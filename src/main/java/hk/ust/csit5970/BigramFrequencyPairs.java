package hk.ust.csit5970;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.commons.cli.*;

import java.io.IOException;
import java.util.HashMap;

public class BigramFrequencyPairs {
    public static class BigramMapper extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private final static IntWritable one = new IntWritable(1);

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase().replaceAll("[^a-z ]", " ").trim();
            String[] words = line.split("\\s+");
            if (words.length < 2) return;

            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].isEmpty() || words[i + 1].isEmpty()) continue;
                context.write(new PairOfStrings(words[i], words[i + 1]), one);
                context.write(new PairOfStrings(words[i], "*"), one);
            }
        }
    }

    public static class BigramCombiner extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            context.write(key, new IntWritable(sum));
        }
    }

    public static class BigramReducer extends Reducer<PairOfStrings, IntWritable, Text, DoubleWritable> {
        private HashMap<String, Integer> firstWordTotals = new HashMap<>();
        private HashMap<String, HashMap<String, Integer>> bigramCounts = new HashMap<>();

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            String w1 = key.getLeftElement();
            String w2 = key.getRightElement();
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            if (w2.equals("*")) {
                firstWordTotals.put(w1, sum);
            } else {
                HashMap<String, Integer> counts = bigramCounts.get(w1);
                if (counts == null) {
                    counts = new HashMap<>();
                    bigramCounts.put(w1, counts);
                }
                counts.put(w2, sum);
            }
        }

        @Override
        protected void cleanup(Context context) throws IOException, InterruptedException {
            for (String w1 : firstWordTotals.keySet()) {
                int total = firstWordTotals.get(w1);
                context.write(new Text(w1), new DoubleWritable((double) total));
                HashMap<String, Integer> bigrams = bigramCounts.get(w1);
                if (bigrams != null) {
                    for (String w2 : bigrams.keySet()) {
                        double freq = (double) bigrams.get(w2) / total;
                        context.write(new Text(w1 + "\t" + w2), new DoubleWritable(freq));
                    }
                }
            }
        }
    }

    public static class BigramPartitioner extends Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numPartitions) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numPartitions;
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Bigram Frequency Pairs");
        job.setJarByClass(BigramFrequencyPairs.class);

        job.setMapperClass(BigramMapper.class);
        job.setCombinerClass(BigramCombiner.class);
        job.setReducerClass(BigramReducer.class);
        job.setPartitionerClass(BigramPartitioner.class);

        job.setMapOutputKeyClass(PairOfStrings.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create("input"));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create("output"));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create("numReducers"));

        CommandLine cmdline;
        try {
            cmdline = new GnuParser().parse(options, args);
        } catch (ParseException e) {
            System.err.println("Error parsing command line: " + e.getMessage());
            new HelpFormatter().printHelp("BigramFrequencyPairs", options);
            System.exit(1);
            return;
        }

        String inputPath = cmdline.getOptionValue("input");
        String outputPath = cmdline.getOptionValue("output");
        int numReducers = cmdline.hasOption("numReducers") ? Integer.parseInt(cmdline.getOptionValue("numReducers")) : 1;

        if (inputPath == null || outputPath == null) {
            System.err.println("Usage: BigramFrequencyPairs -input <input> -output <output> [-numReducers <n>]");
            System.exit(1);
        }

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        job.setNumReduceTasks(numReducers);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
