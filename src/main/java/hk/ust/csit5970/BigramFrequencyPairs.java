package hk.ust.csit5970;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.*;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

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
                PairOfStrings bigram = new PairOfStrings(words[i], words[i + 1]);
                context.write(bigram, one); // Bigram count
                PairOfStrings firstWord = new PairOfStrings(words[i], "*"); // First word total
                context.write(firstWord, one);
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

    public static class BigramReducer extends Reducer<PairOfStrings, IntWritable, Text, FloatWritable> {
        private HashMap<String, Integer> firstWordTotals = new HashMap<String, Integer>();

        @Override
        protected void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            String first = key.getLeftElement();
            String second = key.getRightElement();

            if (second.equals("*")) {
                firstWordTotals.put(first, sum);
                context.write(new Text(first + "\t"), new FloatWritable(sum));
            } else {
                Integer total = firstWordTotals.get(first);
                if (total != null && total > 0) {
                    float frequency = (float) sum / total;
                    context.write(new Text(first + "\t" + second), new FloatWritable(frequency));
                }
            }
        }
    }

    public static class BigramPartitioner extends org.apache.hadoop.mapreduce.Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numPartitions) {
            return Math.abs(key.getLeftElement().hashCode() % numPartitions);
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
        job.setOutputValueClass(FloatWritable.class);

        // Parse command-line arguments
        String inputPath = null;
        String outputPath = null;
        int numReducers = 1; // Default

        for (int i = 0; i < args.length; i++) {
            if ("-input".equals(args[i])) {
                inputPath = args[++i];
            } else if ("-output".equals(args[i])) {
                outputPath = args[++i];
            } else if ("-numReducers".equals(args[i])) {
                numReducers = Integer.parseInt(args[++i]);
            }
        }

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
