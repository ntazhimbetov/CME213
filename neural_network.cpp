#include "neural_network.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms(NeuralNetwork& nn) {
    double norm_sum = 0;

    for(int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu(arma::square(nn.W[i]));
    }

    return norm_sum;
}

void write_cpudata_tofile(NeuralNetwork& nn, int iter) {
    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    nn.W[0].save(s.str(), arma::raw_ascii);
    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    nn.W[1].save(t.str(), arma::raw_ascii);
    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    nn.b[0].save(u.str(), arma::raw_ascii);
    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    nn.b[1].save(v.str(), arma::raw_ascii);
}

void write_diff_gpu_cpu(NeuralNetwork& nn, int iter,
                        std::ofstream& error_file) {
    arma::mat A, B, C, D;

    std::stringstream s;
    s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
    A.load(s.str(), arma::raw_ascii);
    double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
    double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

    std::stringstream t;
    t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
    B.load(t.str(), arma::raw_ascii);
    double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
    double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

    std::stringstream u;
    u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
    C.load(u.str(), arma::raw_ascii);
    double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
    double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

    std::stringstream v;
    v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
    D.load(v.str(), arma::raw_ascii);
    double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
    double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

    int ow = 15;

    if(iter == 0) {
        error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(
                       ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1"
                   << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(
                       ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left
                   << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(
                       ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
    }

    error_file << std::left << std::setw(ow) << iter << std::left << std::setw(
                   ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 <<
               std::left << std::setw(ow) << max_errb0 << std::left << std::setw(
                   ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left <<
               std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 <<
               std::left<< std::setw(ow) << L2_errb1 << "\n";

}

/* CPU IMPLEMENTATIONS */
void feedforward(NeuralNetwork& nn, const arma::mat& X, struct cache& cache) {
    cache.z.resize(2);
    cache.a.resize(2);

    // std::cout << W[0].n_rows << "\n";tw
    assert(X.n_rows == nn.W[0].n_cols);
    cache.X = X;
    int N = X.n_cols;

    arma::mat z1 = nn.W[0] * X + arma::repmat(nn.b[0], 1, N);
    cache.z[0] = z1;

    arma::mat a1;
    sigmoid(z1, a1);
    cache.a[0] = a1;

    assert(a1.n_rows == nn.W[1].n_cols);
    arma::mat z2 = nn.W[1] * a1 + arma::repmat(nn.b[1], 1, N);
    cache.z[1] = z2;

    arma::mat a2;
    softmax(z2, a2);
    cache.a[1] = cache.yc = a2;
}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : C x N one-hot column vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop(NeuralNetwork& nn, const arma::mat& y, double reg,
              const struct cache& bpcache, struct grads& bpgrads) {
    bpgrads.dW.resize(2);
    bpgrads.db.resize(2);
    int N = y.n_cols;

    // std::cout << "backprop " << bpcache.yc << "\n";
    arma::mat diff = (1.0 / N) * (bpcache.yc - y);
    bpgrads.dW[1] = diff * bpcache.a[0].t() + reg * nn.W[1];
    bpgrads.db[1] = arma::sum(diff, 1);
    arma::mat da1 = nn.W[1].t() * diff;

    arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

    bpgrads.dW[0] = dz1 * bpcache.X.t() + reg * nn.W[0];
    bpgrads.db[0] = arma::sum(dz1, 1);
}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss(NeuralNetwork& nn, const arma::mat& yc, const arma::mat& y,
            double reg) {
    int N = yc.n_cols;
    double ce_sum = -arma::accu(arma::log(yc.elem(arma::find(y == 1))));

    double data_loss = ce_sum / N;
    double reg_loss = 0.5 * reg * norms(nn);
    double loss = data_loss + reg_loss;
    // std::cout << "Loss: " << loss << "\n";
    return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict(NeuralNetwork& nn, const arma::mat& X, arma::rowvec& label) {
    struct cache fcache;
    feedforward(nn, X, fcache);
    label.set_size(X.n_cols);

    for(int i = 0; i < X.n_cols; ++i) {
        arma::uword row;
        fcache.yc.col(i).max(row);
        label(i) = row;
    }
}

/*
 * Computes the numerical gradient
 */
void numgrad(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
             double reg, struct grads& numgrads) {
    double h = 0.00001;
    struct cache numcache;
    numgrads.dW.resize(nn.num_layers);
    numgrads.db.resize(nn.num_layers);

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.dW[i].resize(nn.W[i].n_rows, nn.W[i].n_cols);

        for(int j = 0; j < nn.W[i].n_rows; ++j) {
            for(int k = 0; k < nn.W[i].n_cols; ++k) {
                double oldval = nn.W[i](j,k);
                nn.W[i](j, k) = oldval + h;
                feedforward(nn, X, numcache);
                double fxph = loss(nn, numcache.yc, y, reg);
                nn.W[i](j, k) = oldval - h;
                feedforward(nn, X, numcache);
                double fxnh = loss(nn, numcache.yc, y, reg);
                numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
                nn.W[i](j, k) = oldval;
            }
        }
    }

    for(int i = 0; i < nn.num_layers; ++i) {
        numgrads.db[i].resize(nn.b[i].n_rows, nn.b[i].n_cols);

        for(int j = 0; j < nn.b[i].size(); ++j) {
            double oldval = nn.b[i](j);
            nn.b[i](j) = oldval + h;
            feedforward(nn, X, numcache);
            double fxph = loss(nn, numcache.yc, y, reg);
            nn.b[i](j) = oldval - h;
            feedforward(nn, X, numcache);
            double fxnh = loss(nn, numcache.yc, y, reg);
            numgrads.db[i](j) = (fxph - fxnh) / (2*h);
            nn.b[i](j) = oldval;
        }
    }
}

/*
 * Train the neural network &nn
 */
void train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
           double learning_rate, double reg,
           const int epochs, const int batch_size, bool grad_check, int print_every,
           int debug) {
    int N = X.n_cols;
    int iter = 0;
    int print_flag = 0;

    for(int epoch = 0 ; epoch < epochs; ++epoch) {
        int num_batches = (N + batch_size - 1)/batch_size;

	for (int batch = 0; batch < num_batches; ++batch) {
            int last_col = std::min((batch + 1)*batch_size-1, N-1);
            arma::mat X_batch = X.cols(batch * batch_size, last_col);
            arma::mat y_batch = y.cols(batch * batch_size, last_col);

            struct cache bpcache;
            feedforward(nn, X_batch, bpcache);

            struct grads bpgrads;
            backprop(nn, y_batch, reg, bpcache, bpgrads);

            if(print_every > 0 && iter % print_every == 0) {
                if(grad_check) {
                    struct grads numgrads;
                    numgrad(nn, X_batch, y_batch, reg, numgrads);
                    assert(gradcheck(numgrads, bpgrads));
                }

                std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" <<
                          epochs << " = " << loss(nn, bpcache.yc, y_batch, reg) << "\n";
            }

            // Gradient descent step
            for(int i = 0; i < nn.W.size(); ++i) {
                nn.W[i] -= learning_rate * bpgrads.dW[i];
            }

            for(int i = 0; i < nn.b.size(); ++i) {
                nn.b[i] -= learning_rate * bpgrads.db[i];
            }

            /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
               for the first batch of each epoch to avoid saving too many large files.
               Note that for the first time, you have to run debug and serial modes together.
               This will run the following function and write out files to CPUmats folder.
               In the later runs (with same parameters), you can use just the debug flag to
               output diff b/w CPU and GPU without running CPU version */
            if(print_every <= 0) {
                print_flag = batch == 0;
            } else {
                print_flag = iter % print_every == 0;
            }

            if(debug && print_flag) {
                write_cpudata_tofile(nn, iter);
            }

            iter++;
        }
    }
}

/*
 * TODO
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation
 * should mainly be in this function.
 */
void parallel_train(NeuralNetwork& nn, const arma::mat& X, const arma::mat& y,
                    double learning_rate, double reg,
                    const int epochs, const int batch_size, bool grad_check, int print_every,
                    int debug) {

    int rank, num_procs;
    MPI_SAFE_CALL(MPI_Comm_size(MPI_COMM_WORLD, &num_procs));
    MPI_SAFE_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));

    int N = (rank == 0)?X.n_cols:0;
    MPI_SAFE_CALL(MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

    std::ofstream error_file;
    error_file.open("Outputs/CpuGpuDiff.txt");
    int print_flag = 0;
    int iter = 0;


    //////////////////////////////////////////
    //   Allocating memory for CPU and GPU  //
    //////////////////////////////////////////

    int N_special = batch_size / num_procs + num_procs - 1;
    int num_batches = (N + batch_size - 1) / batch_size;

    // Host Memory
    double* X_host = (double*) malloc(sizeof(double) * nn.H[0] * N_special * num_batches);
    double* y_host = (double*) malloc(sizeof(double) * nn.H[2] * N_special * num_batches);
    arma::mat W2 = arma::mat(nn.H[2], nn.H[1]);
    arma::mat W1 = arma::mat(nn.H[1], nn.H[0]);
    arma::mat b2 = arma::mat(nn.H[2], 1);
    arma::mat b1 = arma::mat(nn.H[1], 1);

    arma::mat W2_grad = arma::mat(nn.H[2], nn.H[1]);
    arma::mat W1_grad = arma::mat(nn.H[1], nn.H[0]);
    arma::mat b2_grad = arma::mat(nn.H[2], 1);
    arma::mat b1_grad = arma::mat(nn.H[1], 1);

    // Device Memory
    double* X_dev;      double* y_dev;
    double* W2_dev;     double* W1_dev;
    double* b1_dev;     double* b2_dev;

    double* a2_exp;        double* a1_dev;     double* z2_dev;
    double* W2_dev_copy;   double* diff;

    cudaMalloc((void**)&X_dev, sizeof(double) * nn.H[0] * N_special * num_batches);
    cudaMalloc((void**)&y_dev, sizeof(double) * nn.H[2] * N_special * num_batches);

    cudaMalloc((void**)&W2_dev, sizeof(double) * nn.H[2] * nn.H[1]);
    cudaMalloc((void**)&W1_dev, sizeof(double) * nn.H[1] * nn.H[0]);
    cudaMalloc((void**)&b2_dev, sizeof(double) * nn.H[2]);
    cudaMalloc((void**)&b1_dev, sizeof(double) * nn.H[1]);

    cudaMalloc((void**)&a1_dev, sizeof(double) * nn.H[1] * N_special);
    cudaMalloc((void**)&z2_dev, sizeof(double) * nn.H[2] * N_special);
    cudaMalloc((void**)&a2_exp, sizeof(double) * nn.H[0] * N_special);

    cudaMalloc((void**)&W2_dev_copy, sizeof(double) * nn.H[2] * nn.H[1]);
    cudaMalloc((void**)&diff, sizeof(double) * N_special);

    ///////////////////////////////////////////////
    //          SCATTERING THE DATA              //
    ///////////////////////////////////////////////

    int Xcount = 0;
    int ycount = 0;
    std::vector<int> Xsendcounts(num_procs, 0);
    std::vector<int> Xdispl(num_procs, 0);
    std::vector<int> ysendcounts(num_procs, 0);
    std::vector<int> ydispl(num_procs, 0);

    for(int batch = 0; batch < num_batches; ++batch) {
      int column  = std::min(N - 1, (batch + 1) * batch_size - 1);
      int batch_N = column + 1 - batch * batch_size;
      int N_pc2   = batch_N / num_procs;

      Xsendcounts[0] = (N_pc2 + batch_N % num_procs) * nn.H[0];
      Xdispl[0]      = 0;
      ysendcounts[0] = (N_pc2 + batch_N % num_procs) * nn.H[2];
      ydispl[0]      = 0;

      Xcount = N_pc2 * nn.H[0];   ycount = N_pc2 * nn.H[2];

      if (rank == 0) {
        Xcount += (batch_N % num_procs) * nn.H[0];
        ycount += (batch_N % num_procs) * nn.H[2];
      }

      for(int i = 0; i < num_procs; ++i){
	       Xsendcounts[i] = N_pc2 * nn.H[0];
         Xdispl[i]      = Xsendcounts[i-1] + Xdispl[i-1];
         ysendcounts[i] = N_pc2 * nn.H[2];
         ydispl[i]      = ysendcounts[i-1] + ydispl[i-1];
      }

      MPI_SAFE_CALL(MPI_Scatterv(X.colptr(batch * batch_size),
				 Xsendcounts.data(), Xdispl.data(), MPI_DOUBLE,
				 X_host + batch * (N_special) * nn.H[0], Xcount,
                                 MPI_DOUBLE, 0, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Scatterv(y.colptr(batch * batch_size),
				 ysendcounts.data(), ydispl.data(), MPI_DOUBLE,
				 y_host + batch * (N_special) * nn.H[2], ycount,
                                 MPI_DOUBLE, 0, MPI_COMM_WORLD));
    }

    // Copying the data from host to the device
    cudaMemcpy(X_dev,
	       X_host,
	       sizeof(double) * nn.H[0] * N_special * num_batches,
	       cudaMemcpyHostToDevice);

    cudaMemcpy(y_dev,
	       y_host,
	       sizeof(double) * nn.H[2] * N_special * num_batches,
	       cudaMemcpyHostToDevice);


    for(int epoch = 0; epoch < epochs; ++epoch) {
      for(int batch = 0; batch < num_batches; ++batch) {
        // Copying from the Host to Device

        cudaMemcpy(W1_dev, nn.W[0].memptr(), sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyHostToDevice);
        cudaMemcpy(W2_dev, nn.W[1].memptr(), sizeof(double) * nn.H[1] * nn.H[2], cudaMemcpyHostToDevice);
        cudaMemcpy(b1_dev, nn.b[0].memptr(), sizeof(double) * nn.H[1], cudaMemcpyHostToDevice);
        cudaMemcpy(b2_dev, nn.b[1].memptr(), sizeof(double) * nn.H[2], cudaMemcpyHostToDevice);

        // Defining the constants and important batch sizes
        int column = std::min(N - 1, (batch + 1) * batch_size - 1);
        int batch_N = column + 1 - batch * batch_size;
        int N_pc2 = batch_N / num_procs;
        // irregularity update
        if (rank == 0) {
          N_pc2 += batch_N % num_procs;
        }


        ///////////////////////////////////////////
        //            FEED FORWARD               //
        ///////////////////////////////////////////

        GEMMSigmoid(W1_dev, X_dev + batch * N_special * nn.H[0], b1_dev, a1_dev, 1.0, 1.0, nn.H[1], N_pc2, nn.H[0]);
        GEMMAddition(W2_dev, a1_dev, b2_dev, z2_dev, 1.0, 1.0, nn.H[2], N_pc2, nn.H[1]);
        SumOfExpCol(z2_dev, a2_exp, nn.H[2], N_pc2);
        Softmax(z2_dev, a2_exp, y_dev + N_special * batch * nn.H[2], diff, 1.0/N_pc2, nn.H[2], N_pc2);

        cudaMemcpy(W2_dev_copy,
                   W2_dev,
                   sizeof(double) * nn.H[1] * nn.H[2],
                   cudaMemcpyDeviceToDevice);

        ///////////////////////////////////////////
        //              BACK PROP                //
        ///////////////////////////////////////////

        GEMMTranspose(diff, a1_dev, W2_dev_copy, reg, nn.H[2], nn.H[1], N_pc2);
        SumOfRow(diff, b2_dev, nn.H[2], N_pc2);
        Hadamard(W2_dev, diff, a1_dev, nn.H[1], N_pc2, nn.H[2]);
        GEMMTranspose(a1_dev, X_dev + batch * N_special * nn.H[0], W1_dev, reg, nn.H[1], nn.H[0], N_pc2);
        SumOfRow(a1_dev, b1_dev, nn.H[1], N_pc2);


        // Copying from the Device to the Host
        cudaMemcpy(W2.memptr(), W2_dev_copy, sizeof(double) * nn.H[1] * nn.H[2], cudaMemcpyDeviceToHost);
        cudaMemcpy(W1.memptr(), W1_dev, sizeof(double) * nn.H[0] * nn.H[1], cudaMemcpyDeviceToHost);
        cudaMemcpy(b2.memptr(), b2_dev, sizeof(double) * nn.H[2], cudaMemcpyDeviceToHost);
        cudaMemcpy(b1.memptr(), b1_dev, sizeof(double) * nn.H[1], cudaMemcpyDeviceToHost);

	    ///////////////////////////////////////////
        //              REDUCTION                //
        ///////////////////////////////////////////

        MPI_SAFE_CALL(MPI_Allreduce(W2.memptr(),
                      W2_grad.memptr(),
                      nn.H[1] * nn.H[2],
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(W1.memptr(),
                      W1_grad.memptr(),
                      nn.H[0] * nn.H[1],
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(b2.memptr(),
                      b2_grad.memptr(),
                      nn.H[2],
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD));
        MPI_SAFE_CALL(MPI_Allreduce(b1.memptr(),
                      b1_grad.memptr(),
                      nn.H[1],
                      MPI_DOUBLE,
                      MPI_SUM,
                      MPI_COMM_WORLD));

	      ///////////////////////////////////////////
	      //            GRADIENT DESCENT           //
	      ///////////////////////////////////////////

        nn.W[0] = nn.W[0] - (learning_rate / num_procs) * W1_grad;
        nn.W[1] = nn.W[1] - (learning_rate / num_procs) * W2_grad;
        nn.b[0] = nn.b[0] - (learning_rate / num_procs) * b1_grad;
        nn.b[1] = nn.b[1] - (learning_rate / num_procs) * b2_grad;

        if(print_every <= 0) {
          print_flag = batch == 0;
        } else {
          print_flag = iter % print_every == 0;
        }

        /* Following debug routine assumes that you have already updated the arma
        matrices in the NeuralNetwork nn.  */
        if(debug && rank == 0 && print_flag) {
          write_diff_gpu_cpu(nn, iter, error_file);
        }

        iter++;
      }
    }

    // Freeing the memory

    cudaFreeHost(X_host);
    cudaFreeHost(y_host);

    cudaFree(X_dev);
    cudaFree(y_dev);
    cudaFree(W2_dev);
    cudaFree(W1_dev);
    cudaFree(b1_dev);
    cudaFree(b2_dev);
    cudaFree(a2_exp);
    cudaFree(a1_dev);
    cudaFree(z2_dev);
    cudaFree(W2_dev_copy);
    cudaFree(diff);

    error_file.close();
}
