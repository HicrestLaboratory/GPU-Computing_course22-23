# Pinned memory, streams and cuda events

Exercise 1: find the better kernel dimensions

1. Taking back the vector add exercise done in the lab 2, choose the layout (kernel layout 1 or kernel layout 2), block size and grid size that perform better and put them on a new file that use streams pinned memory and events (take the lecture “Asynchronous concurrent executions” as reference).
2. Measure the time time to solution with CUDA events and the kernel bandwidth.
3. Iterate over the stream configurations (like the number of streams used) to find the best configuration.
4. Compare them with the original solution taken by lab 2
