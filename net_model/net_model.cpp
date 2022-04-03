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
#include <omp.h>
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

double h(double x) {
    return -sqrt(2*fabs(log(x)));
}

double H(double x) {
    return exp(-(x*x)/2.)/(sqrt(2*M_PI)*fabs(x));
}

class Cell {

    public:
    bool active;
    vector<int> pre;
    int i,k; 

    Cell(double act, double i2, double k2) {
        active = act;
        i = i2;
        k = k2;
    }

    void update() {
        double u=0.;
        for (int j = 0; j < pre.size(); j++) {
            u = u + J[k][population[pre[j]].k]*population[pre[j]].active;
        }
        u = u + ext[k] - theta[k];
        active = heaviside(u);
    }
};

extern "C" int simulate(int N_E, int N_I, int K, int T, double* exp_e, double* exp_i) {
    
    int N;
    double m_0, E, I, J_EE, J_IE, J_E, J_I;
    N=N_E+N_I;

    m_0=0.3;
    E=1.;
    I=0.8;
    J_EE=1.;
    J_IE=1.;
    J_E=2.;
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
    bernoulli_distribution distribution3(probs[1]);
    
    for (int i = 0; i < N; i++) {
        vector<int> nds;
        for (int j = 0; j < N_E; ++j) {
            if (distribution2(generator)) nds.push_back(j);
        }
        for (int j = 0; j < N_I; ++j) {
            if (distribution3(generator)) nds.push_back(N_E+j);
        }
        population[i].pre = nds;
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
    //printf("Initial number of excitatory active cells is %.2f\n", (double)n_active_e/N_E*100);
    //printf("Initial number of inhibitory active cells is %.2f\n", (double)n_active_i/N_I*100);
    
    
    uniform_int_distribution<> distr(0, N_E-1);
    int obs = distr(generator);

    ofstream me_file ("me.txt");
    ofstream mi_file ("mi.txt");
    ofstream se_file ("spikes_e.txt");
    ofstream si_file ("spikes_i.txt");
    ofstream obs_ex_file ("obs_ex.txt");
    ofstream obs_in_file ("obs_in.txt");
    ofstream obs_spikes_file ("obs_spikes.txt");
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    int idx, before, after;
    double ue, ui;
    me_file << n_active_e << '\n';
    mi_file << n_active_i << '\n';
    for (int i = 0; i < T; i++) {
        random_shuffle(indices.begin(), indices.end());
        for (int j = 0; j < N; j++) {
            idx = indices[j];
            before = population[idx].active;
            population[idx].update();
            after = population[idx].active;
            if (idx == obs) {
                ue=0; ui=0;
                for (int j = 0; j < population[idx].pre.size(); j++) {
                    if (population[population[idx].pre[j]].k == 0) {
                        ue = ue + J[population[idx].k][population[population[idx].pre[j]].k]*population[population[idx].pre[j]].active;
                    }
                }
                ue = ue + ext[population[idx].k];
                obs_ex_file << ue << '\n';

                for (int j = 0; j < population[idx].pre.size(); j++) {
                    if (population[population[idx].pre[j]].k == 1) {
                        ui = ui + J[population[idx].k][population[population[idx].pre[j]].k]*population[population[idx].pre[j]].active;
                    }
                }
                obs_in_file << ui << '\n';
                if (after-before == 1) {
                    obs_spikes_file << i << '\n';
                }
            }
            if (after-before == 1) {
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

    //printf("Final number of excitatory active cells is %.2f\n", (double)n_active_e/N_E*100);
    //printf("Final number of inhibitory active cells is %.2f\n", (double)n_active_i/N_I*100);
    
    *exp_e = (J_I*E-J_E*I)/(J_E-J_I)*m_0;
    *exp_i = (E-I)/(J_E-J_I)*m_0;
    //printf("Expected excitatory active cells is %.2f\n", *exp_e*100);
    //printf("Expected inhibitory active cells is %.2f\n", *exp_i*100);

    /*
    printf("FINITE K CORRECTIONS:\n");

    double fKc_e, fKc_i, alpha_e, alpha_i, me, mi;
    me = (*exp_e);  mi = (*exp_i);
    alpha_e = me+mi*J_E*J_E;
    alpha_i = me+mi*J_I*J_I;
    fKc_e = (E*m_0+me-J_E*mi)-(theta[0]+sqrt(alpha_e)*h(me))/sqrt(K);
    fKc_i = (I*m_0+me-J_I*mi)-(theta[1]+sqrt(alpha_i)*h(mi))/sqrt(K);

    printf("%f\n",fKc_e);
    printf("%f\n",fKc_i);
    */

    vector<Cell>().swap(population);
    vector<double>().swap(ext);
    vector<double>().swap(theta);
    vector<double>().swap(probs);
    return 1;
}
