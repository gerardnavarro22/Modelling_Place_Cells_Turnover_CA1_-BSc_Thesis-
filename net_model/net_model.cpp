// Program to illustrate the working of
// public and private in C++ Class

#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <algorithm>
#include <cstring>
#include <list>
#include <fstream>
using namespace std;


class Cell;
vector<vector<double>> J(2, vector<double>(2));
vector<Cell> population;
vector<double> ext;
vector<double> theta;
vector<double> probs;

bool heaviside(double x) {
    if (x<=0) {
        return false;
    } else {
        return true;
    }
}

double dist() {
        default_random_engine generator(time(0));
        bernoulli_distribution distribution(probs[0]);
        return distribution(generator);
    }

class Cell {

   /*private:
    double something;*/

    public:
    bool active;
    vector<int> pre;
    int i,k; 


    // function to initialize private variables
    Cell(double act, double i2, double k2) {
        active = act;
        i = i2;
        k = k2;
    }

    void update() {
        double u=0;
        for (int j = 0; j < pre.size(); j++) {
            u = u + J[k][population[pre[j]].k]*population[pre[j]].active;
        }
        u = u + ext[k] - theta[k];
        active = heaviside(u);
    }
};

extern "C" int simulate(int N_E, int N_I, int K, double* exp_e, double* exp_i) {
    int N;
    double m_0, E, I, J_EE, J_IE, J_E, J_I;
    N=N_E+N_I;

    m_0=0.3;
    E=1.;
    I=0.8;
    J_EE=1.;
    J_IE=1.;
    J_E=2;
    J_I=1.8;

    J[0][0] = J_EE/sqrt(K);
    J[0][1] = -J_E/sqrt(K);
    J[1][0] = J_IE/sqrt(K);
    J[1][1] = -J_I/sqrt(K);

    if (!((E/I) < (-J[0][1]/-J[1][1])) && !((E/I) > (-J[0][1]/-J[1][1]))) {
        if (!((-J[0][1]/-J[1][1]) < 1.) && !((-J[0][1]/-J[1][1]) > 1.)) {
            cout << "Bad params";
            exit(-1);
        }
    }
    ext = {E*m_0*sqrt(K), I*m_0*sqrt(K)};
    theta = {1., 0.7};
    probs = {(double)K/N_E, (double)K/N_I};
    
    default_random_engine generator(time(0));
    bernoulli_distribution distribution(0.5);
    for (int i = 0; i < N_E; i++) {
        population.push_back(Cell(distribution(generator), i, 0));
    }
    for (int i = 0; i < N_I; i++) {
        population.push_back(Cell(distribution(generator), i, 1));
    }
    
    bernoulli_distribution distribution2(probs[0]);
    
    for (int i = 0; i < N_E; i++) {
        vector<int> nds;
        for (int j = 0; j < N; ++j) {
            if (distribution2(generator)) nds.push_back(j);
        }
        population[i].pre = nds;
        vector<int>().swap(nds);
    }

    bernoulli_distribution distribution3(probs[1]);
    for (int i = 0; i < N_I; i++) {
        vector<int> nds;
        for (int j = 0; j < N; ++j) {
            if (distribution3(generator)) nds.push_back(j);
        }
        population[N_E+i].pre = nds;
        vector<int>().swap(nds);
    }
    
    int n_active_e = 0, n_active_i = 0;
    for (int i = 0; i < N; i++) {
        if (population[i].k == 0) {
            n_active_e = n_active_e + population[i].active;
        } else {
            n_active_i = n_active_i + population[i].active;
        }
    }
    printf("Initial number of excitatory active cells is %.2f\n", (double)n_active_e/N_E*100);
    printf("Initial number of inhibitory active cells is %.2f\n", (double)n_active_i/N_I*100);
    
    
    ofstream me_file ("me(t).txt");
    ofstream mi_file ("mi(t).txt");
    ofstream se_file ("spikes_e(t).txt");
    ofstream si_file ("spikes_i(t).txt");
    uniform_int_distribution<> distr(0, N-1);
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    const int T=250;
    //int *spikes = new int[2*T];
    //memset(spikes, 0, sizeof *spikes * 2 * T);
    int idx, before, after;
    me_file << n_active_e << '\n';
    mi_file << n_active_i << '\n';
    for (int i = 0; i < T; i++) {
        random_shuffle(indices.begin(), indices.end());
        for (int j = 0; j < N; j++) {
            //idx = distr(generator);
            idx = indices[j];
            before = population[idx].active;
            population[idx].update();
            after = population[idx].active;
            if (after-before == 1) {
                //spikes[population[idx].k*T+idx] = 1;
                if (population[idx].k == 0) {
                    se_file << population[idx].i << ' ';
                    n_active_e++;
                } else {
                    si_file << population[idx].i << ' ';
                    n_active_i++;
                }
            } else if (after-before == -1) {
                if (population[idx].k == 0) {
                    n_active_e--;
                } else {
                    n_active_i--;
                }
            }
        }
        se_file << '\n';
        si_file << '\n';
        me_file << n_active_e << '\n';
        mi_file << n_active_i << '\n';
    }

    /*
    const int T=250*N;
    int *spikes = new int[2*T];
    memset(spikes, 0, sizeof *spikes * 2 * T);
    int idx, before, after;
    for (int t = 0; t < T; t++) {
        idx = distr(generator);
        before = population[idx].active;
        population[idx].update();
        after = population[idx].active;
        if (after-before == 1) {
            spikes[population[idx].k*T+t] = 1;
            if (population[idx].k == 0) {
                se_file << population[idx].i << ' ';
                n_active_e++;
            } else {
                si_file << population[idx].i << ' ';
                n_active_i++;
            }
        } else if (after-before == -1) {
            if (population[idx].k == 0) {
                n_active_e--;
            } else {
                n_active_i--;
            }
        }
        me_file << n_active_e << '\n';
        mi_file << n_active_i << '\n';
    }
    */


    printf("Final number of excitatory active cells is %.2f\n", (double)n_active_e/N_E*100);
    printf("Final number of inhibitory active cells is %.2f\n", (double)n_active_i/N_I*100);
    
    *exp_e = (J_I*E-J_E*I)/(J_E-J_I)*m_0;
    *exp_i = (E-I)/(J_E-J_I)*m_0;
    printf("Expected excitatory active cells is %.2f\n", *exp_e*100);
    printf("Expected inhibitory active cells is %.2f\n", *exp_i*100);

    //delete[] spikes;
    vector<Cell>().swap(population);
    //vector<vector<double>>().swap(J);
    vector<double>().swap(ext);
    vector<double>().swap(theta);
    vector<double>().swap(probs);
    return 1;
}
