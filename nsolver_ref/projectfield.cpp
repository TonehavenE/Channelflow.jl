/**
 * This file is a part of channelflow version 2.0.
 * License is GNU GPL version 2 or later: https://channelflow.org/license
 */
#include "basisfunctions.h"
#include "cfbasics/cfvector.h"
#include "cfbasics/mathdefs.h"
#include "channelflow/diffops.h"
#include "channelflow/flowfield.h"
#include "channelflow/periodicfunc.h"
#include "channelflow/utilfuncs.h"
#include "polynomial.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdlib.h>
#include <sys/stat.h>

using namespace std;
using namespace chflow;

int main(int argc, char *argv[]) {
  cfMPI_Init(&argc, &argv);
  {
    string purpose("project velocity field onto a given symmetric subspace)\n");
    ArgList args(argc, argv, purpose);

    // const Real   umagn    = args.getreal("-m", "--magnitude", 0.0, "nonzero
    // => rescale u to have this magnitude"); const string symmname =
    // args.getstr(3, "<symmetries>", "symmetry group filename");

    const bool normalize =
        args.getflag("-nrm", "--normalize", "normalize basis elements");
    const bool savebasis = args.getflag("-sb", "--savebasis",
                                        "output basis elements as flowfields");
    const bool saveIP =
        args.getflag("-sIP", "--saveIPmat", "output the inner product matrix");
    const bool plotbasis = args.getflag(
        "-pb", "--plotbasis", "save slices of basis elements for plotting");
    const bool printnorms = args.getflag(
        "-pn", "--printnorms", "print L2Norm(psi[n]), L2Norm(div(u)), etc.");
    const string plotdir =
        args.getstr("-o", "--plotdir", "plots/", "directory for basis plots");
    const string xfilename = args.getstr(
        "-x", "--xfilename", "", "build u = sum psi_n x_n using x from file");
    const string Afilename = args.getstr("-A", "--Afilename", "",
                                         "load inner product matrix from file");
    const string ijklname = args.getstr(
        3, "<ijklfile>", "filename for ijkl indices of basis functions");
    const string uinname =
        args.getstr(2, "<infield>",
                    "input field filename (fully-resolved u to be projected)");
    const string outlabel = args.getstr(
        1, "<outlabel>",
        "filename stub for outputs e.g. foo, produces ufoo.nc and xfoo.asc");
    args.check();
    args.save();

    FlowField u(uinname);
    u.makeSpectral();

    Real Nx = u.Nx();
    Real Ny = u.Ny();
    Real Nz = u.Nz();
    Real Lx = u.Lx();
    Real Lz = u.Lz();
    Real ua = u.a(); // should check these are -1,1
    Real ub = u.b();
    Real alpha = 2 * pi / Lx;
    Real gamma = 2 * pi / Lz;

    cout << setprecision(16);
    cout << "alpha, gamma == " << alpha << ", " << gamma << endl;
    cout << "Nx, Ny, Nz == " << Nx << ", " << Ny << ", " << Nz << endl;

    cout << "Reading ijkl indices of basis set from file" << endl;
    MatrixXi ijkl = readijkl(ijklname);
    cout << "ijkl[0] == " << ijkl(0, 0) << ' ' << ijkl(0, 1) << ' '
         << ijkl(0, 2) << ' ' << ijkl(0, 3) << endl;

    // Figure out the largest value of L in index set
    int L = 0;
    for (int n = 0; n < ijkl.rows(); ++n)
      L = ijkl(n, 3) > L ? ijkl(n, 3) : L;

    cout << "L == max l == " << L << endl;

    cout << "Constructing Legendre polynomials" << endl;
    P.resize(L + 1);
    cout << "resized, trying to make first polynomial" << endl;
    cout << P << endl;
    P[0] = Polynomial(1.0);
    cout << "made first polynomial" << endl;
    if (L >= 1)
      P[1] = Polynomial(0.0, 1.0);
    for (int l = 2; l <= L; ++l)
      P[l] = ((2 * l - 1.0) / l) * P[1] * P[l - 1] - ((l - 1.0) / l) * P[l - 2];

    cout << "Print Legendre polynomials ..." << endl;
    for (int l = 0; l <= L; ++l)
      cout << "P[" << l << "] == " << P[l] << endl;

    cout << "Constructing S, and Sprime polynomials" << endl;

    Polynomial smasher(1.0, 0.0, -1.0);
    Polynomial smasher2 = smasher * smasher;
    Polynomial zeropoly = 0.0 * smasher;

    S.resize(L + 1);
    Sprime.resize(L + 1);

    S[0] = Polynomial(0.0, 1.0, 0.0, -1.0 / 3.0); // S0(y) = y - y^3/3
    cout << "2S[0] == " << 2 * S[0] << endl;
    for (int l = 1; l <= L; ++l) {
      S[l] = P[l - 1] * smasher2;
      // cout << "2S[" << l << "] == " << 2*S[l] << endl;
    }
    for (int l = 0; l <= L; ++l) {
      Sprime[l] = diff(S[l]);
      // cout << "2Sprime[" << l << "] == " << 2*Sprime[l] << endl;
    }

    Vector xgrid = periodicpoints(Nx, Lx);
    Vector ygrid = chebypoints(Ny, ua, ub);
    Vector zgrid = periodicpoints(Nz, Lz);

    // build H-symmetric fundamental basis functions Psi(i,j,k,l) using i,j,k,l,
    // indices (generated in Julia) from a file.

    // A wasteful way to generate / evaluate basis set as FlowFields. Could
    // evaluate on low-order grid then expand with FlowField::resize. Will see
    // how long this takes for 50 basis functions

    int Npsi = ijkl.rows();
    cfarray<FlowField> psi(Npsi);

    // Evaluate Psi(i,j,k,l,m, x,y,z) over grid and assign to flowfield psi[n]
    // where ijkl[n,:] = [i,j,k,l]

    for (int n = 0; n < Npsi; ++n) {
      cout << "Generating psi[" << n + 1 << "] == " << flush;
      psi[n] = FlowField(Nx, Ny, Nz, 3, Lx, Lz, ua, ub);
      psi[n].setState(Physical, Physical);

      int i = ijkl(n, 0);
      int j = ijkl(n, 1);
      int k = ijkl(n, 2);
      int l = ijkl(n, 3);
      cout << "psi(" << i << ", " << j << ", " << k << ", " << l << ")" << endl;
      // m == 0,1,2 is the index of the components of psi vector
      for (int m = 0; m < 3; ++m) {
        for (int nx = 0; nx < Nx; ++nx) {
          Real x = xgrid[nx];
          for (int nz = 0; nz < Nz; ++nz) {
            Real z = zgrid[nz];
            for (int ny = 0; ny < Ny; ++ny) {
              Real y = ygrid[ny];
              psi[n](nx, ny, nz, m) = Psi(i, j, k, l, m, x, y, z, alpha, gamma);
            }
          }
        }
      }

      psi[n].makeSpectral();
      if (normalize)
        psi[n] *= 1.0 / L2Norm(psi[n]);

      if (savebasis)
        psi[n].save("psi" +
                    i2s(n + 1)); // n+1 in order to match Julia 1-based indices
    }
    cout << endl;

    if (printnorms) {
      for (int n = 0; n < Npsi; ++n)
        cout << "L2Norm(Ψ[" << n + 1 << "]) == " << L2Norm(psi[n]) << endl;
      cout << endl;

      cout << "Computing norms of basis..." << endl;
      for (int n = 0; n < Npsi; ++n)
        cout << "L2Norm(Ψ[" << n + 1 << "]) == " << L2Norm(psi[n]) << endl;
      cout << endl;

      cout << "Computing divergence of basis..." << endl;
      for (int n = 0; n < Npsi; ++n) {
        psi[n].makeSpectral();
        cout << "L2Norm(div(Ψ[" << n + 1 << "])) == " << divNorm(psi[n])
             << endl;
      }
      cout << endl;
    }

    cout << "Computing innerproduct of u onto psi[1] to psi[" << Npsi << "] ..."
         << endl;
    Eigen::VectorXd ip_u_psi(Npsi);
    for (int i = 0; i < Npsi; ++i)
      ip_u_psi[i] = L2IP(psi[i], u);

    if (plotbasis) {
      cout << "Saving psi[n] plot data..." << flush;
      Vector x = periodicpoints(u.Nx(), u.Lx());
      Vector y = chebypoints(u.Ny(), u.a(), u.b());
      Vector z = periodicpoints(u.Nz(), u.Lz());
      ofstream osx((plotdir + "x.asc").c_str());
      ofstream osy((plotdir + "y.asc").c_str());
      ofstream osz((plotdir + "z.asc").c_str());
      for (int nx = 0; nx <= u.Nx(); ++nx)
        osx << x(nx) << '\n';
      for (int ny = 0; ny < u.Ny(); ++ny)
        osy << y(ny) << '\n';
      for (int nz = 0; nz <= u.Nz(); ++nz)
        osz << z(nz) << '\n';

      for (int n = 0; n < Npsi; ++n)
        plotfield(0.1 * psi[n], plotdir, string("psi") + i2s(n + 1), 1, 1, 1);
      cout << endl;
    }

    Eigen::VectorXd a;
    if (xfilename == "") {
      Eigen::MatrixXd IP(Npsi, Npsi);

      if (Afilename == "") {
        cout << "Computing inner product matrix..." << endl;
        for (int m = 0; m < Npsi; ++m) {
          cout << m << ' ' << flush;
          int mj = ijkl(m, 1); // fourier j,k indices of Psi[m]
          int mk = ijkl(m, 2);
          for (int n = 0; n < m; ++n) {
            int nj = ijkl(n, 1); // fourier j,k indices of Psi[n]
            int nk = ijkl(n, 2);

            // nonmatching Fourier modes are orthogonal. Short-circuit eval
            // those IPs to zero.
            Real ip = (abs(mj) == abs(nj) && abs(mk) == abs(nk))
                          ? L2IP(psi[m], psi[n])
                          : 0.0;
            ip = (fabs(ip) > 1e-15) ? ip : 0.0;
            IP(m, n) = ip;
            IP(n, m) = ip;
          }
          IP(m, m) = L2IP(psi[m], psi[m]);
        }
        cout << endl;
        if (saveIP)
          save(IP, "IP");

        // cout << "innerproduct matrix == \n" << IP << endl;
      } else
        load(IP, Afilename);

      cout << "Solving for coefficients of u in psi basis..." << flush;
      a = IP.partialPivLu().solve(ip_u_psi);

      cout << "Saving expansion to file..." << flush;
      string xoutname = string("x") + outlabel;
      save(a, xoutname);

    } else {
      cout << "Loading expansion coeffs from file..." << flush;
      load(a, xfilename);
    }

    cout << "Expansion coeffs = " << endl << a << endl;

    cout << "ijkl indices vs coeff table..." << endl;
    for (int n = 0; n < Npsi; ++n) {
      cout << "n==" << n + 1 << ' ';
      for (int m = 0; m < 4; ++m)
        cout << (ijkl(n, m) < 0 ? '-' : ' ') << abs(ijkl(n, m)) << ' ';
      cout << (a[n] < 0 ? '-' : ' ') << abs(a[n]) << endl;
    }

    cout << "Expanding projected field..." << flush;
    FlowField uout(Nx, Ny, Nz, 3, Lx, Lz, ua, ub);
    uout.makeSpectral();
    for (int i = 0; i < Npsi; ++i)
      uout += a[i] * psi[i];
    cout << endl;

    cout << "L2Norm(u)      == " << L2Norm(u) << endl;
    cout << "L2Norm(uout)   == " << L2Norm(uout) << endl;
    cout << "L2Dist(u,uout) == " << L2Dist(u, uout) << endl;
    cout << "L2Dist(u,uout)/L2Norm(u) == " << L2Dist(u, uout) / L2Norm(u)
         << endl;

    string uoutname = string("u") + outlabel;
    uout.save(uoutname);

    cout << setprecision(3);
    for (int i = 0; i < Npsi; ++i)
      cout << a[i] << "  ";
    cout << endl;
  }
  cfMPI_Finalize();
}

#include "basisfunctions.cpp"
#include "polynomial.cpp"
