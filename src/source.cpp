#define _USE_MATH_DEFINES

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <chrono>

#define R 6378000.0
#define D 200000.0
#define GM 398600.5
#define N 8102
#define EPSILON 0.000000000001

using std::cos;
using std::sin;
using std::sqrt;

int main(int argc, char** argv)
{
    int threads = 1;
    omp_set_num_threads(threads); // set number of threads

    auto start = std::chrono::high_resolution_clock::now();

    double* B = new double[N] {0.0};
    double* L = new double[N] {0.0};
    double Brad = 0.0, Lrad = 0.0, H = 0.0, u2n2 = 0.0;
    double temp = 0.0;

    // suradnice bodov X_i
    double* X_x  = new double[N] {0.0};
    double* X_y  = new double[N] {0.0};
    double* X_z  = new double[N] {0.0};
    double xNorm = 0.0;
    
    // suradnice bodov s_j
    double* s_x = new double[N] {0.0};
    double* s_y = new double[N] {0.0};
    double* s_z = new double[N] {0.0};
    
    // suradnicce normal v x_i
    double* n_x = new double[N] {0.0};
    double* n_y = new double[N] {0.0};
    double* n_z = new double[N] {0.0};
    
    // g vektor
    double* g = new double[N] {0.0};

    // r vector
    double r_x = 0.0;
    double r_y = 0.0;
    double r_z = 0.0;
    double rNorm = 0.0;
    double rNorm3 = 0.0;

    // dot product of vector r with normal n[i]
    double Kij = 0.0;
    int i = 0;
    int j = 0;

    // load data
    FILE* file = nullptr;
    file = fopen("E:/_school/4_ZS/paralelne_algoritmy/tyzden6_TiazovePoleZeme_openMP/build/Debug/BL-8102.dat", "r");
    if (file == nullptr)
    {
        printf("file not open\n");
        return -1;
    }

    for (i = 0; i < N; i++)
    {
        int result = fscanf(file, "%lf %lf %lf %lf %lf", &B[i], &L[i], &H, &g[i], &u2n2);
        g[i] = -g[i] * 0.00001;
        
        //g[i] = u2n2;
        Brad = B[i] * M_PI / 180.0;
        Lrad = L[i] * M_PI / 180.0;

        X_x[i] = R * cos(Brad) * cos(Lrad);
        X_y[i] = R * cos(Brad) * sin(Lrad);
        X_z[i] = R * sin(Brad);

        s_x[i] = (R - D) * cos(Brad) * cos(Lrad);
        s_y[i] = (R - D) * cos(Brad) * sin(Lrad);
        s_z[i] = (R - D) * sin(Brad);
        
        xNorm = sqrt(X_x[i] * X_x[i] + X_y[i] * X_y[i] + X_z[i] * X_z[i]);
        n_x[i] = -X_x[i] / xNorm;
        n_y[i] = -X_y[i] / xNorm;
        n_z[i] = -X_z[i] / xNorm;
        
        //if (i < 10)
          //  printf("g[%d]: %.5lf\n", i, g[i]);
        //    printf("X[%d] = (%.2lf, %.2lf, %.2lf)\n", i, X_x[i], X_y[i], X_z[i]);
    }

    fclose(file);

    //for (int i = 0; i < N; i++) // set constant g values
    //    g[i] = -(GM) / (R * R);

    // vytvorenie matice systemu rovnic
    double* A = new double[N * N] {0.0};
    int ij = -1;

#pragma omp parallel for private(j,r_x,r_y,r_z,rNorm,rNorm3,Kij,ij)
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            // compute vector r & its norm
            r_x = X_x[i] - s_x[j];
            r_y = X_y[i] - s_y[j];
            r_z = X_z[i] - s_z[j];

            rNorm  = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

            rNorm3 = rNorm * rNorm * rNorm;
            
            // dot product of vector r and normal n_i
            Kij = r_x * n_x[i] + r_y * n_y[i] + r_z * n_z[i];
            //if (i == j && i < 10)
            //    printf("Kij: %.4lf\n", Kij);
            
            // compute 
            ij = i * N + j;
            A[ij] = (1.0 / (4.0 * M_PI * rNorm3)) * Kij;
            /*if (i == j && i < 10)
                printf("Kij: %.4lf\n", A[ij]);*/
        }
    }

    //########## BCGS linear solver ##########//

    double* sol = new double[N]; // vektor x^0 -> na ukladanie riesenia systemu
    double* r_hat = new double[N]; // vektor \tilde{r} = b - A.x^0;
    double* r = new double[N]; // vektor pre rezidua
    double* p = new double[N]; // pomocny vektor na update riesenia
    double* v = new double[N]; // pomocny vektor na update riesenia
    double* s = new double[N]; // pomocny vektor na update riesenia
    double* t = new double[N]; // pomocny vektor na update riesenia

    double beta = 0.0;
    double rhoNew = 1.0;
    double rhoOld = 0.0;
    double alpha = 1.0;
    double omega = 1.0;

    double tempDot = 0.0;
    double tempDot2 = 0.0;
    double sNorm = 0.0;

    int MAX_ITER = 1000;
    double TOL = 1.0E-6;
    int iter = 1;

    double rezNorm = 0.0;
    for (i = 0; i < N; i++) // set all to zero
    {
        sol[i] = 0.0;
        p[i] = 0.0; // = 0
        v[i] = 0.0; // = 0
        s[i] = 0.0;
        t[i] = 0.0;

        r[i] = g[i];
        r_hat[i] = g[i];
        rezNorm += r[i] * r[i];

    }

    printf("||r0||: %.10lf\n", sqrt(rezNorm));
    rezNorm = 0.0;

    do
    {
        rhoOld = rhoNew; // save previous rho_{i-2}
        rhoNew = 0.0; // compute new rho_{i-1}
        for (i = 0; i < N; i++) // dot(r_hat, r)
            rhoNew += r_hat[i] * r[i];

        if (rhoNew == 0.0)
            return -1;

        //if (iter == 1)
        //{
        //    printf("iter 1 setup\n");
        //    for (int i = 0; i < N; i++)
        //        p[i] = r[i];
        //}
        //else
        //{
        beta = (rhoNew / rhoOld) * (alpha / omega);
        for (i = 0; i < N; i++) // update vector p^(i)
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
        //}

        // compute vector v = A.p
#pragma omp parallel for private(j,ij)
        for (i = 0; i < N; i++)
        {
            v[i] = 0.0;
            for (j = 0; j < N; j++)
            {
                ij = i * N + j;
                v[i] += A[ij] * p[j];
            }
        }

        // compute alpha
        tempDot = 0.0;
        for (i = 0; i < N; i++)
            tempDot += r_hat[i] * v[i];

        alpha = rhoNew / tempDot;

        // compute vektor s
        sNorm = 0.0;
        for (i = 0; i < N; i++)
        {
            s[i] = r[i] - alpha * v[i];
            sNorm += s[i] * s[i];
        }

        sNorm = sqrt(sNorm);
        if (sNorm < TOL) // check if ||s|| is small enough
        {
            for (i = 0; i < N; i++) // update solution x
                sol[i] = sol[i] + alpha * p[i];

            printf("BCGS stop: ||s|| is small enough, iter: %d\n", iter);
            break;
        }

        // compute vector t = A.s
#pragma omp parallel for private(j,ij)
        for (i = 0; i < N; i++)
        {
            t[i] = 0.0;
            for (j = 0; j < N; j++)
            {
                ij = i * N + j;
                t[i] += A[ij] * s[j];
            }
        }

        // compute omega
        tempDot = 0.0; tempDot2 = 0.0;
        for (i = 0; i < N; i++)
        {
            tempDot += t[i] * s[i];
            tempDot2 += t[i] * t[i];
        }
        omega = tempDot / tempDot2;

        rezNorm = 0.0;
        for (i = 0; i < N; i++)
        {
            sol[i] = sol[i] + alpha * p[i] + omega * s[i]; // update solution x
            r[i] = s[i] - omega * t[i]; // compute new residuum vector
            rezNorm += r[i] * r[i]; // compute residuum norm
        }

        rezNorm = sqrt(rezNorm);
        printf("iter: %d    ||r||: %.10lf\n", iter, rezNorm);

        if (rezNorm < TOL)
        {
            printf("BCGS stop iter: ||r|| is small enough\n");
            break;
        }

        iter++;

    } while ((iter < MAX_ITER) && (rezNorm > TOL));

    delete[] r_hat;
    delete[] r;
    delete[] p;
    delete[] v;
    delete[] s;
    delete[] t;

    //########## EXPORT DATA ##########//
    file = fopen("out.dat", "w");
    if (file == nullptr)
    {
        printf("data export failed\n");
        return -1;
    }

    printf("solution export started... ");
    double ui = 0.0, Gij = 0.0;
    for (i = 0; i < N; i++)
    {
        ui = 0.0;
        for (j = 0; j < N; j++) // compute solution u(X_i)
        {
            r_x = X_x[i] - s_x[j];
            r_y = X_y[i] - s_y[j];
            r_z = X_z[i] - s_z[j];

            rNorm = sqrt(r_x * r_x + r_y * r_y + r_z * r_z);

            Gij = 1.0 / (4.0 * M_PI * rNorm);

            ui += sol[j] * Gij;
        }

        fprintf(file, "%.5lf\t%.5lf\t%.5lf\n", B[i], L[i], ui);
    }

    fclose(file);
    printf("done\n");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("\nthreads: %d -> duration: %.4lf seconds\n", threads, (double)duration.count() / 1000000.0);

    delete[] X_x;
    delete[] X_y;
    delete[] X_z;
    delete[] s_x;
    delete[] s_y;
    delete[] s_z;
    delete[] n_x;
    delete[] n_y;
    delete[] n_z;
    delete[] g;
    delete[] A;
    delete[] sol;
}