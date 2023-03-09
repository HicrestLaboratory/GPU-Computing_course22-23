# Laboratory I: exercise2

You have to write a C code that, given an integer n in input, generates two random vectors a and b of length 2n and then computes the vector c = a + b with his computation time. After this you have to compute the mean and the variance of a, b, and c.
Before writing your solution code inside the macro with "Put here your code" in files 'makefile', 'lab1_ex2.c', 'src/lab1_e2_lib.c' and 'include/lab1_e2_lib.h', you will use this exercise also to practice with git/GitHub and makefiles. Therefore, to correctly complete this exercise:

1. Fork the repository at https://github.com/HicrestLaboratory/GPU-Computing_course22-23 on your GitHub account.

2. Download the repository from your GitHub with the command:
'git clone https://github.com/YOUR_GITHUB_ACCOUNT/YOUR_FORK_NAME'
or (if you installed the github cli) with:
'gh repo clone YOUR_GITHUB_ACCOUNT/YOUR_FORK_NAME'

3. Open the makefile and complete it as reported in the comments; to check, you have to launch 'make' and then './bin/lab1_ex2 3' and obtain the following output:

        argv[0] = 3
        n = 3
        dtype = int
        v name |      mu(v) |   sigma(v) |
                a |   0.000000 |   0.000000 |
                b |   0.000000 |   0.000000 |
                c |   0.000000 |   0.000000 |

        MEAN TEST (|mu(c) - (mu(a) + mu(b))| < 0.001?):          ERROR!

4. Open the file 'lab1_ex2.c' and complete it as reported in the comments inside its; at the end, the last test must return "DONE!" instead of "ERROR!"
