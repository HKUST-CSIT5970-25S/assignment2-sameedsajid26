package hk.ust.csit5970;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.Arrays;

public class CORPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORPairs.class);

    private static class CORMapper1 extends Mapper<LongWritable, Text, Text, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            HashMap<String, Integer> word_set = new HashMap<>();
            String clean_doc = value.toString().replaceAll("[^a-z A-Z]", " ").toLowerCase();
            StringTokenizer doc_tokenizer = new StringTokenizer(clean_doc);

            // Count word frequencies (all occurrences)
            while (doc_tokenizer.hasMoreTokens()) {
                String word = doc_tokenizer.nextToken();
                Integer count = word_set.get(word);
                word_set.put(word, (count == null ? 0 : count) + 1);
            }
            for (Map.Entry<String, Integer> entry : word_set.entrySet()) {
                context.write(new Text("W:" + entry.getKey()), new IntWritable(entry.getValue()));
            }

            // Count bigrams (once per line)
            Set<String> bigrams = new HashSet<>();
            String[] words = clean_doc.split("\\s+");
            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].isEmpty() || words[i + 1].isEmpty()) continue;
                String w1 = words[i];
                String w2 = words[i + 1];
                String bigram = w1.compareTo(w2) < 0 ? w1 + "\t" + w2 : w2 + "\t" + w1;
                bigrams.add(bigram);
            }
            for (String bigram : bigrams) {
                context.write(new Text("B:" + bigram), one);
            }
        }
    }

    private static class CORReducer1 extends Reducer<Text, IntWritable, Text, IntWritable> {
        @Override
        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class CORPairsMapper2 extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().trim();
            System.err.println("Mapper2 processing line: [" + line + "]");
            String[] parts = line.split("\t");
            System.err.println("Split parts: " + Arrays.toString(parts));
            if (parts.length != 3) {
                System.err.println("Invalid mid-result line, expected 3 parts: " + line);
                return;
            }
            if (parts[0].startsWith("B:")) {
                String w1 = parts[0].substring(2);
                String w2 = parts[1];
                int count;
                try {
                    count = Integer.parseInt(parts[2]);
                    context.write(new PairOfStrings(w1, w2), new IntWritable(count));
                    System.err.println("Emitted: (" + w1 + ", " + w2 + ") -> " + count);
                } catch (NumberFormatException e) {
                    System.err.println("Invalid count in line: " + line);
                }
            } else {
                System.err.println("Line doesn’t start with B:: " + line);
            }
        }
    }

    private static class CORPairsCombiner2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class CORPairsReducer2 extends Reducer<PairOfStrings, IntWritable, PairOfStrings, DoubleWritable> {
        private final static Map<String, Integer> word_total_map = new HashMap<>();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            Path middle_result_path = new Path("mid/part-r-00000");
            Configuration middle_conf = context.getConfiguration();
            try {
                FileSystem fs = FileSystem.get(URI.create(middle_result_path.toString()), middle_conf);
                if (!fs.exists(middle_result_path)) {
                    LOG.error("Middle result path does not exist: " + middle_result_path);
                    return;
                }

                FSDataInputStream in = fs.open(middle_result_path);
                InputStreamReader inStream = new InputStreamReader(in);
                BufferedReader reader = new BufferedReader(inStream);

                LOG.info("Reading middle result...");
                String line = reader.readLine();
                while (line != null) {
                    String[] line_terms = line.split("\t");
                    if (line_terms.length == 2 && line_terms[0].startsWith("W:")) {
                        word_total_map.put(line_terms[0].substring(2), Integer.parseInt(line_terms[1]));
                    }
                    line = reader.readLine();
                }
                reader.close();
                LOG.info("Loaded " + word_total_map.size() + " word frequencies");
            } catch (Exception e) {
                LOG.error("Error reading middle result: " + e.getMessage());
            }
        }

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            LOG.info("Reducer processing key: " + key);
            int freqAB = 0;
            for (IntWritable val : values) {
                freqAB += val.get();
            }
            LOG.info("freqAB for " + key + ": " + freqAB);

            String w1 = key.getLeftElement();
            String w2 = key.getRightElement();
            Integer freqA = word_total_map.get(w1);
            Integer freqB = word_total_map.get(w2);

            if (freqA != null && freqB != null && freqA > 0 && freqB > 0) {
                double cor = (double) freqAB / (freqA * freqB);
                context.write(new PairOfStrings(w1, w2), new DoubleWritable(cor));
                LOG.info("Output: " + w1 + "\t" + w2 + "\t" + cor);
            } else {
                LOG.warn("Skipping COR for " + w1 + ", " + w2 + ": freqA=" + freqA + ", freqB=" + freqB);
            }
        }
    }

    private static class MyPartitioner extends Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    public CORPairs() {}

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @SuppressWarnings("static-access")
    public int run(String[] args) throws Exception {
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();
        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(CORPairs.class.getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String middlePath = "mid";
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + CORPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - middle path: " + middlePath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "CORPairs First Pass");
        job1.setJarByClass(CORPairs.class);
        job1.setMapperClass(CORMapper1.class);
        job1.setReducerClass(CORReducer1.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(IntWritable.class);
        FileInputFormat.setInputPaths(job1, new Path(inputPath));
        FileOutputFormat.setOutputPath(job1, new Path(middlePath));
        FileSystem.get(conf1).delete(new Path(middlePath), true);
        long startTime = System.currentTimeMillis();
        boolean success = job1.waitForCompletion(true);
        LOG.info("Job 1 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");
        if (!success) return 1;

        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "CORPairs Second Pass");
        job2.setJarByClass(CORPairs.class);
        job2.setMapperClass(CORPairsMapper2.class);
        job2.setCombinerClass(CORPairsCombiner2.class);
        job2.setReducerClass(CORPairsReducer2.class);
        job2.setPartitionerClass(MyPartitioner.class);
        job2.setMapOutputKeyClass(PairOfStrings.class);
        job2.setMapOutputValueClass(IntWritable.class);
        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setNumReduceTasks(reduceTasks);
        FileInputFormat.setInputPaths(job2, new Path(middlePath));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath));
        FileSystem.get(conf2).delete(new Path(outputPath), true);
        startTime = System.currentTimeMillis();
        success = job2.waitForCompletion(true);
        LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new CORPairs(), args);
    }
}
