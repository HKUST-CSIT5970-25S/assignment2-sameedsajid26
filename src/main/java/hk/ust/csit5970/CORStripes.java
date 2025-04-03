package hk.ust.csit5970;

import org.apache.commons.cli.*;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
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
import java.util.*;

public class CORStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(CORStripes.class);

    // First-pass Mapper: Count individual word frequencies
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
        }
    }

    // First-pass Reducer: Sum word frequencies
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

    // Second-pass Mapper: Build stripes of co-occurrences
    public static class CORStripesMapper2 extends Mapper<LongWritable, Text, Text, MapWritable> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String doc_clean = value.toString().replaceAll("[^a-z A-Z]", " ").toLowerCase();
            StringTokenizer doc_tokenizers = new StringTokenizer(doc_clean);
            List<String> words = new ArrayList<>();
            Set<String> seen_bigrams = new HashSet<>(); // Track bigrams per line

            // Collect words in the line
            while (doc_tokenizers.hasMoreTokens()) {
                words.add(doc_tokenizers.nextToken());
            }

            // Build stripes for each word
            for (int i = 0; i < words.size(); i++) {
                String w1 = words.get(i);
                MapWritable stripe = new MapWritable();
                for (int j = 0; j < words.size(); j++) {
                    if (i == j) continue; // Skip self
                    String w2 = words.get(j);
                    String bigram = w1.compareTo(w2) < 0 ? w1 + "\t" + w2 : w2 + "\t" + w1;
                    if (!seen_bigrams.contains(bigram)) {
                        stripe.put(new Text(w2), new IntWritable(1));
                        seen_bigrams.add(bigram);
                    }
                }
                if (!stripe.isEmpty()) {
                    context.write(new Text(w1), stripe);
                }
            }
        }
    }

    // Second-pass Combiner: Aggregate stripes
    public static class CORStripesCombiner2 extends Reducer<Text, MapWritable, Text, MapWritable> {
        @Override
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            MapWritable combinedStripe = new MapWritable();
            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    Text word = (Text) entry.getKey();
                    IntWritable count = (IntWritable) entry.getValue();
                    IntWritable existing = (IntWritable) combinedStripe.get(word);
                    int sum = (existing == null ? 0 : existing.get()) + count.get();
                    combinedStripe.put(word, new IntWritable(sum));
                }
            }
            context.write(key, combinedStripe);
        }
    }

    // Second-pass Reducer: Compute COR using word frequencies from mid-result
    public static class CORStripesReducer2 extends Reducer<Text, MapWritable, PairOfStrings, DoubleWritable> {
        private static Map<String, Integer> word_total_map = new HashMap<>();

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
        protected void reduce(Text key, Iterable<MapWritable> values, Context context) throws IOException, InterruptedException {
            String w1 = key.toString();
            MapWritable combinedStripe = new MapWritable();

            // Aggregate all stripes for w1
            for (MapWritable stripe : values) {
                for (Map.Entry<Writable, Writable> entry : stripe.entrySet()) {
                    Text w2 = (Text) entry.getKey();
                    IntWritable count = (IntWritable) entry.getValue();
                    IntWritable existing = (IntWritable) combinedStripe.get(w2);
                    int sum = (existing == null ? 0 : existing.get()) + count.get();
                    combinedStripe.put(w2, new IntWritable(sum));
                }
            }

            // Compute COR for each w2 in the stripe
            Integer freqA = word_total_map.get(w1);
            if (freqA == null || freqA == 0) return;

            for (Map.Entry<Writable, Writable> entry : combinedStripe.entrySet()) {
                String w2 = ((Text) entry.getKey()).toString();
                int freqAB = ((IntWritable) entry.getValue()).get();
                Integer freqB = word_total_map.get(w2);

                if (freqB != null && freqB > 0) {
                    double cor = (double) freqAB / (freqA * freqB);
                    context.write(new PairOfStrings(w1, w2), new DoubleWritable(cor));
                }
            }
        }
    }

    public CORStripes() {}

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
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String middlePath = "mid";
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + CORStripes.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - middle path: " + middlePath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        // First Pass
        Configuration conf1 = new Configuration();
        Job job1 = Job.getInstance(conf1, "CORStripes First Pass");
        job1.setJarByClass(CORStripes.class);
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

        // Second Pass
        Configuration conf2 = new Configuration();
        Job job2 = Job.getInstance(conf2, "CORStripes Second Pass");
        job2.setJarByClass(CORStripes.class);
        job2.setMapperClass(CORStripesMapper2.class);
        job2.setCombinerClass(CORStripesCombiner2.class);
        job2.setReducerClass(CORStripesReducer2.class);
        job2.setMapOutputKeyClass(Text.class);
        job2.setMapOutputValueClass(MapWritable.class);
        job2.setOutputKeyClass(PairOfStrings.class);
        job2.setOutputValueClass(DoubleWritable.class);
        job2.setNumReduceTasks(reduceTasks);
        FileInputFormat.setInputPaths(job2, new Path(inputPath));
        FileOutputFormat.setOutputPath(job2, new Path(outputPath));
        FileSystem.get(conf2).delete(new Path(outputPath), true);
        startTime = System.currentTimeMillis();
        success = job2.waitForCompletion(true);
        LOG.info("Job 2 Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return success ? 0 : 1;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new CORStripes(), args);
    }
}
