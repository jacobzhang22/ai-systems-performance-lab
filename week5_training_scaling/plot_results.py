import csv
import matplotlib.pyplot as plt


batch_sizes = []
throughputs = []
memories = []

with open("results.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        batch_sizes.append(int(row["batch_size"]))
        throughputs.append(float(row["throughput_samples_per_s"]))
        memories.append(float(row["peak_memory_mb"]))

plt.figure()
plt.plot(batch_sizes, throughputs, marker="o")
plt.xlabel("Batch size")
plt.ylabel("Throughput (samples/s)")
plt.title("Training Throughput vs Batch Size")
plt.grid(True)
plt.savefig("throughput_vs_batch.png", bbox_inches="tight")

plt.figure()
plt.plot(batch_sizes, memories, marker="o")
plt.xlabel("Batch size")
plt.ylabel("Peak GPU memory (MB)")
plt.title("Peak GPU Memory vs Batch Size")
plt.grid(True)
plt.savefig("memory_vs_batch.png", bbox_inches="tight")