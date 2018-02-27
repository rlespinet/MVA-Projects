#include <iostream>
#include <Eigen/Dense>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <random>

#include "RedSVD.h"

using namespace Eigen;

struct PGM {

    ~PGM() {
        if (data != NULL) {
            delete[] data;
        }
    }
    uint32_t width;
    uint32_t height;
    uint8_t *data;
};

int readpgm(PGM &pgm, const char *path) {

    std::ifstream infile(path);

    if (infile)
    {

        char buffer[128];
        infile.getline(buffer, 128);
        if (!std::strcmp(buffer, "P5\n")) return 1;

        int width, height;
        infile >> width;
        infile >> height;
        infile.ignore(16, '\n');
        infile.ignore(16, '\n');

        uint8_t *data = new uint8_t[width * height + 1];

        infile.read((char*)(data), width * height);
        data[width * height] = '\0';


        pgm.width = width;
        pgm.height = height;
        pgm.data = data;

        return 0;
    }
    else
    {
        return 1;
    }

}

int writepgm(PGM &pgm, const char* path) {

    std::ofstream outfile(path);

    if (outfile) {

        outfile << "P5\n";
        outfile << pgm.width << " " << pgm.height << '\n';
        outfile << "255\n";
        outfile.write((char*) pgm.data, pgm.width * pgm.height);

    } else {
        return 1;
    }

    return 0;
}

// def lrmc(X, W, tau = 1000, beta = 2, max_iters = 100, A_init=None):
//     A = np.empty_like(X)
//     Z = np.zeros_like(X) if A_init is None else beta * (X - A_init) * W
//     for i in range(max_iters):
//         A = SVT(Z * W, tau)
//         S = beta * (X - A) * W
//         Z += S
//         if np.max(np.abs(S)) < 0.1:
//             break
//     print("iterations : ", i+1)

//     return A

MatrixXd svt(const MatrixXd &X, double tau) {

    BDCSVD<MatrixXd> svd(X.sparseView(), ComputeThinU | ComputeThinV);
    VectorXd s = svd.singularValues();
    // std::cout << "s size : " << s.size() << std::endl;
    for (int i = 0; i < s.size(); i++) {
        s(i) = std::max(s(i) - tau, 0.0);
    }
    // std::cout << "s:" << s << std::endl;
    return svd.matrixU() * (s.asDiagonal() * svd.matrixV().transpose());
}

MatrixXd lrmc(const MatrixXd &X, const MatrixXd &W, double tau, double beta, uint32_t max_iters) {

    MatrixXd Z = MatrixXd::Zero(X.rows(), X.cols());
    MatrixXd A(X.rows(), X.cols());
    MatrixXd S(X.rows(), X.cols());

    MatrixXd Z_proj(X.rows(), X.cols());

    for (int i = 0; i < max_iters; i++) {
	if (!(i % 500)) std::cout << i << "/" << max_iters << std::endl;
	A = svt(Z.array() * W.array(), tau);
        S = beta * (X.array() - A.array()) * W.array();
        Z = Z + S;
        if (S.norm() < 0.1) {
            break;
        }
    }

    return A;

}

void fill_bernoulli(MatrixXd &W, float rho) {
    static std::default_random_engine generator (1);
    std::bernoulli_distribution distribution(rho);

    for (int x = 0; x < W.cols(); x++) {
        for (int y = 0; y < W.rows(); y++) {
            W(y, x) = distribution(generator) ? 1.0 : 0.0;
        }
    }

}

// MatrixXd corrupt(const MatrixXd& X, float rho) {

//     static std::default_random_engine generator (1);

//     MatrixXd X_corrupt(X);

//     std::bernoulli_distribution distribution(1.0f - rho);
//     for (int x = 0; x < X.cols(); x++) {
//         for (int y = 0; y < X.rows(); y++) {
//             X_corrupt(y, x) = (distribution(generator) ? X(y, x) : 0);
//         }
//     }

//     return X_corrupt;
// }

int main(int ac, char **av) {


    std::ifstream list("../../data/images/list_reduced.info");

    std::vector<std::string> files;
    while (list) {
        std::string file;
        list >> file;
        list.ignore(16, '\n');
        files.push_back(file);
    }

    MatrixXd X;
    uint32_t im_width, im_height;

    for (int i = 0; i < files.size(); i++) {

        PGM pgm;

        std::string path = std::string("../../data/images/") + files[i];
        // path = std::string("test.pgm");

        int result = readpgm(pgm, path.c_str());
        if (result) {
            std::cout << "Error could not read file " << path << std::endl;
            return 1;
        }

        if (X.size() == 0) {
            X.resize(pgm.width * pgm.height, files.size());
            im_width = pgm.width;
            im_height = pgm.height;
        }

        for (int j = 0; j < pgm.width * pgm.height; j++) {
            X(j, i) = static_cast<double>(pgm.data[j]);
        }

    }

    MatrixXd W(X.rows(), X.cols());
    fill_bernoulli(W, 1.0 - 0.8);
    // MatrixXd M = MatrixXd::Constant(W.rows(), W.cols(), 1.0) - W;

    MatrixXd X_corrupt = X.array() * W.array();

    MatrixXd X_recover = lrmc(X_corrupt, W, 100000, 2, 2000);

    VectorXd V_recover = X_recover.col(1);
    // VectorXd V_recover = X_corrupt.col(ac);

    double V_max = V_recover.maxCoeff();
    double V_min = V_recover.minCoeff();

    PGM pgm = {im_width, im_height, new uint8_t[im_height * im_width + 1]};
    pgm.data[im_height * im_width] = '\0';

    for (int i = 0; i < V_recover.size(); i++) {
        double coeff = (V_recover(i) - V_min) / (V_max - V_min) * 255.0;
        // std::cout << static_cast<int>(static_cast<unsigned char>(coeff)) << " ";
        pgm.data[i] = static_cast<unsigned char>(coeff);
    }
    // std::cout << std::endl;

    writepgm(pgm, "output.pgm");

    return 0;
}
