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
import java.util.Iterator;
import java.util.Map;

public class BigramFrequencyStripes {
    public static class BigramMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
        private HashMapStringIntWritable stripe = new HashMapStringIntWritable();

        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String line = value.toString().toLowerCase().replaceAll("[^a-z ]", " ").trim();
            String[] words = line.split("\\s+");
            if (words.length < 2) return;

            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].isEmpty() || words[i + 1].isEmpty()) continue;
                stripe.clear();
                stripe.put(words[i + 1], 1); // Bigram
                context.write(new Text(words[i]), stripe);
                stripe.clear();
                stripe.put("*", 1); // Word count
                context.write(new Text(words[i]), stripe);
            }
        }
    }

    public static class BigramCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        private HashMapStringIntWritable resultStripe = new HashMapStringIntWritable();

        @Override
        protected void reduce(Text key, Iterable<HashMapStringIntWritable> values, Context context) throws IOException, InterruptedException {
            resultStripe.clear();
            Iterator<HashMapStringIntWritable> iter = values.iterator();
            while (iter.hasNext()) {
                HashMapStringIntWritable stripe = iter.next();
                Iterator<Map.Entry<String, Integer>> entries = stripe.entrySet().iterator();
                while (entries.hasNext()) {
                    Map.Entry<String, Integer> entry = entries.next();
                    String w2 = entry.getKey();
                    int count = entry.getValue();
                    Integer existing = resultStripe.get(w2);
                    if (existing == null) {
                        resultStripe.put(w2, count);
                    } else {
                        resultStripe.put(w2, existing + count);
                    }
                }
            }
            context.write(key, resultStripe);
        }
    }

    public static class BigramReducer extends Reducer<Text, HashMapStringIntWritable, Text, DoubleWritable> {
        private HashMapStringIntWritable totalStripe = new HashMapStringIntWritable();

        @Override
        protected void reduce(Text key, Iterable<HashMapStringIntWritable> values, Context context) throws IOException, InterruptedException {
            totalStripe.clear();
            int totalCount = 0;

            Iterator<HashMapStringIntWritable> iter = values.iterator();
            while (iter.hasNext()) {
                HashMapStringIntWritable stripe = iter.next();
                Iterator<Map.Entry<String, Integer>> entries = stripe.entrySet().iterator();
                while (entries.hasNext()) {
                    Map.Entry<String, Integer> entry = entries.next();
                    String w2 = entry.getKey();
                    int count = entry.getValue();
                    if (w2.equals("*")) {
                        totalCount += count;
                    } else {
                        Integer existing = totalStripe.get(w2);
                        if (existing == null) {
                            totalStripe.put(w2, count);
                        } else {
                            totalStripe.put(w2, existing + count);
                        }
                    }
                }
            }

            context.write(new Text(key.toString()), new DoubleWritable((double) totalCount));
            Iterator<Map.Entry<String, Integer>> totalEntries = totalStripe.entrySet().iterator();
            while (totalEntries.hasNext()) {
                Map.Entry<String, Integer> entry = totalEntries.next();
                double frequency = (double) entry.getValue() / totalCount;
                context.write(new Text(key.toString() + "\t" + entry.getKey()), new DoubleWritable(frequency));
            }
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Bigram Frequency Stripes");
        job.setJarByClass(BigramFrequencyStripes.class);

        job.setMapperClass(BigramMapper.class);
        job.setCombinerClass(BigramCombiner.class);
        job.setReducerClass(BigramReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);

        String inputPath = null;
        String outputPath = null;
        int numReducers = 1;

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
            System.err.println("Usage: BigramFrequencyStripes -input <input> -output <output> [-numReducers <n>]");
            System.exit(1);
        }

        FileInputFormat.addInputPath(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));
        job.setNumReduceTasks(numReducers);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
