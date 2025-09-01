#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <deque>
#include <random>

void generate_data(float *x, const unsigned long n, const unsigned long m) {
    std::random_device rd;
    std::mt19937 engine(rd());

    std::uniform_real_distribution<float> dist(0.0, 1.0);
    for (unsigned long i = 0; i < n*m; i++) x[i] = dist(engine);
}

namespace extension_cpp {
    struct Matrix {
        float *data;
        unsigned long n;
        unsigned long m;
    };

    struct MatrixCost {
        unsigned long cost;
        unsigned long n;
        unsigned long m;
        unsigned int break_pt;
    };

    void get_dot(const Matrix &a, const Matrix &b, Matrix &c) {
        for (unsigned long i = 0; i < a.n; i++) {
            for (unsigned long j = 0; j < b.m; j++) {
                c.data[i*b.m + j] = 0.0;;
            }
        }

        for (unsigned long i = 0; i < a.n; i++) {
            for (unsigned long j = 0; j < a.m; j++) {
                for (unsigned long k = 0; k < b.m; k++) {
                    c.data[i*b.m + k] += a.data[i*a.m + j]*b.data[j*b.m + k];
                }
            }
        }
    }

    void copy_matrix(const Matrix &src, Matrix &dst) {
        unsigned long n = src.n;
        unsigned long m = src.m;
        dst.data = new float[n*m];
        dst.n = n;
        dst.m = m;
        for (unsigned long k = 0; k < n*m; k++) dst.data[k] = src.data[k];
    }

    void dot_chain(const Matrix *inp, float *out, const unsigned int num_matrices) {
        MatrixCost *dp = new MatrixCost[num_matrices*num_matrices];
        unsigned int *next_idx = new unsigned int[num_matrices];
        for (unsigned int i = 0; i < num_matrices; i++) next_idx[i] = i;

        for (unsigned int length = 1; length <= num_matrices; length++) {
            for (unsigned int i = 0; i < num_matrices-length+1; i++) {
                unsigned int j = i+length-1;

                if (length == 1) dp[i*num_matrices + j] = {0, inp[i].n, inp[i].m, i};           
                else if (length == 2) dp[i*num_matrices + j] = {inp[i].n*inp[i].m*inp[j].m, inp[i].n, inp[j].m, j};
                else {
                    unsigned long min_cost = LONG_MAX;
                    unsigned int min_n = 0;
                    unsigned int min_m = 0;
                    unsigned int min_break_pt = i;

                    for (unsigned int k = i; k < j; k++) {
                        MatrixCost a = dp[i*num_matrices + k];
                        MatrixCost b = dp[(k+1)*num_matrices + j];

                        unsigned long new_cost = a.cost + b.cost + a.n*a.m*b.m;
                        if (new_cost < min_cost) {
                            min_cost = new_cost;
                            min_n = a.n;
                            min_m = b.m;
                            min_break_pt = k;
                        }
                    }

                    dp[i*num_matrices + j] = {min_cost, min_n, min_m, min_break_pt};
                }
            }
        }

        std::cout << dp[num_matrices-1].cost << std::endl;

        std::deque<std::pair<unsigned int, unsigned int>> q;
        q.push_back(std::make_pair(0, num_matrices-1));

        while (q.size() > 0) {
            std::pair<unsigned int, unsigned int> z = q.front();
            q.pop_front();

            unsigned int i = z.first;
            unsigned int j = z.second;

            if (i == j or i == j-1) next_idx[i] = j;
            else {
                MatrixCost a = dp[i*num_matrices + j];

                if (a.break_pt >= i && a.break_pt < j) {
                    q.push_back(std::make_pair(i, a.break_pt));
                    q.push_back(std::make_pair(a.break_pt+1, j));
                }
            }
        }

        Matrix d = {nullptr, 0, 0};

        unsigned int i = 0;

        while (i < num_matrices) {
            unsigned int j = next_idx[i];
            if (i == j) {
                if (d.n == 0) {
                    copy_matrix(inp[i], d);
                }
                else {
                    unsigned long n = d.n;
                    unsigned long m = inp[i].m;
                    Matrix d1;
                    d1.data = new float[n*m];
                    d1.n = n;
                    d1.m = m;
                    get_dot(d, inp[i], d1);
                    copy_matrix(d1, d);
                }   
            }
            else if (i == j-1) {
                unsigned long n = inp[i].n;
                unsigned long m = inp[j].m;

                Matrix c;
                c.data = new float[n*m];
                c.n = n;
                c.m = m;
                get_dot(inp[i], inp[j], c);
                
                if (d.n == 0) {
                    copy_matrix(c, d);
                }
                else {
                    n = d.n;
                    m = c.m;
                    Matrix d1;
                    d1.data = new float[n*m];
                    d1.n = n;
                    d1.m = m;
                    get_dot(d, c, d1);
                    copy_matrix(d1, d);
                }
            }

            i = j+1;
        }

        if (d.n != 0) for (unsigned long k = 0; k < d.n*d.m; k++) out[k] = d.data[k];
    }
}

void print_vector(float *x, size_t n) {
    std::cout << "[";
    for (auto i = 0; i < n; i++) std::cout << x[i] << ", ";
    std::cout << "]" << std::endl;
}

void populate_matrix(extension_cpp::Matrix &matrix, const unsigned long n, const unsigned long m) {
    matrix.n = n;
    matrix.m = m;
    matrix.data = new float[n*m];
    generate_data(matrix.data, n, m);
}


int main(int argc, char *argv[]) {
    unsigned int num_matrices = 3;
    extension_cpp::Matrix *matrices = new extension_cpp::Matrix[num_matrices];

    populate_matrix(matrices[0], 10, 20);
    populate_matrix(matrices[1], 20, 5);
    populate_matrix(matrices[2], 5, 10);

    float *out = new float[10*10];
    extension_cpp::dot_chain(matrices, out, num_matrices);
    print_vector(out, 100);

    populate_matrix(matrices[0], 10, 3);
    populate_matrix(matrices[1], 3, 5);
    populate_matrix(matrices[2], 5, 10);

    out = new float[10*10];
    extension_cpp::dot_chain(matrices, out, num_matrices);
    print_vector(out, 100);
}
