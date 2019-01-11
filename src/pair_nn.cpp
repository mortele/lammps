/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Paul Crozier (SNL)
------------------------------------------------------------------------- */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <iomanip>
#include <armadillo>
#include "pair_nn.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace MathConst;
using namespace std;;
using namespace arma;

/* ---------------------------------------------------------------------- */

PairNN::PairNN(LAMMPS *lmp) : Pair(lmp) {
  respa_enable = 1;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairNN::~PairNN() {
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(epsilon);
    memory->destroy(sigma);
    memory->destroy(lj1);
    memory->destroy(lj2);
    memory->destroy(lj3);
    memory->destroy(lj4);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairNN::compute(int eflag, int vflag) {
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r2inv,r6inv,forcelj,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  int     i0 = 2412;
  double Ei0 = 0;
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    tagint itag = tag[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      tagint jtag = tag[j];

      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {
        //r2inv = 1.0/rsq;
        //r6inv = r2inv*r2inv*r2inv;
        //forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
        //fpair = factor_lj*forcelj*r2inv;

        double r = sqrt(rsq);
        double E = network(r);
        double F = backPropagation();
        fpair = -F/r;
        
        //printf("%30.15g   %30.15g\n", r, (-F)/r - fpair);
        //printf("%30.15g   %30.15g\n", r, E - 4*(1/pow(r,12) + 1/pow(r,6))+ 4*(1/pow(2.5,12)-1/pow(2.5,6)));
    
        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          //evdwl = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) - offset[itype][jtype];
          //evdwl *= factor_lj;;
          evdwl = E;
          if (itag==i0 || jtag==i0) {
            //std::cout << "hei" << std::endl;
            //fflush(stdout);
            Ei0 += E;
          }
          //std::cout << std::setprecision(20) << sqrt(rsq) << " " << fabs(E-evdwl) << std::endl;
          //exit(1);
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }
  //if (evflag) outFile << std::setprecision(20) << Ei0 << std::endl;

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairNN::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(epsilon,n+1,n+1,"pair:epsilon");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(lj1,n+1,n+1,"pair:lj1");
  memory->create(lj2,n+1,n+1,"pair:lj2");
  memory->create(lj3,n+1,n+1,"pair:lj3");
  memory->create(lj4,n+1,n+1,"pair:lj4");
  memory->create(offset,n+1,n+1,"pair:offset");
  load("");
  //outFile.open("Ei0.txt",std::ios::out);
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNN::settings(int narg, char **arg) {
  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNN::coeff(int narg, char **arg) {
  if (narg < 4 || narg > 5)
    error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR,arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR,arg[1],atom->ntypes,jlo,jhi);

  double epsilon_one = force->numeric(FLERR,arg[2]);
  double sigma_one = force->numeric(FLERR,arg[3]);

  double cut_one = cut_global;
  if (narg == 5) cut_one = force->numeric(FLERR,arg[4]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      epsilon[i][j] = epsilon_one;
      sigma[i][j] = sigma_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNN::init_style()
{
  // request regular or rRESPA neighbor lists

  int irequest;

  if (update->whichflag == 1 && strstr(update->integrate_style,"respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;

    if (respa == 0) irequest = neighbor->request(this,instance_me);
    else if (respa == 1) {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->respaouter = 1;
    } else {
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 1;
      neighbor->requests[irequest]->respainner = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 2;
      neighbor->requests[irequest]->respamiddle = 1;
      irequest = neighbor->request(this,instance_me);
      neighbor->requests[irequest]->id = 3;
      neighbor->requests[irequest]->respaouter = 1;
    }

  } else irequest = neighbor->request(this,instance_me);

  // set rRESPA cutoffs

  if (strstr(update->integrate_style,"respa") &&
      ((Respa *) update->integrate)->level_inner >= 0)
    cut_respa = ((Respa *) update->integrate)->cutoff;
  else cut_respa = NULL;
}

/* ----------------------------------------------------------------------
   neighbor callback to inform pair style of neighbor list to use
   regular or rRESPA
------------------------------------------------------------------------- */

void PairNN::init_list(int id, NeighList *ptr)
{
  if (id == 0) list = ptr;
  else if (id == 1) listinner = ptr;
  else if (id == 2) listmiddle = ptr;
  else if (id == 3) listouter = ptr;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairNN::init_one(int i, int j) {
  if (setflag[i][j] == 0) {
    epsilon[i][j] = mix_energy(epsilon[i][i],epsilon[j][j],
                               sigma[i][i],sigma[j][j]);
    sigma[i][j] = mix_distance(sigma[i][i],sigma[j][j]);
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  lj1[i][j] = 48.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj2[i][j] = 24.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  lj3[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],12.0);
  lj4[i][j] = 4.0 * epsilon[i][j] * pow(sigma[i][j],6.0);
  
  if (offset_flag && (cut[i][j] > 0.0)) {
    double ratio = sigma[i][j] / cut[i][j];
    offset[i][j] = 4.0 * epsilon[i][j] * (pow(ratio,12.0) - pow(ratio,6.0));
  } else offset[i][j] = 0.0;

  lj1[j][i] = lj1[i][j];
  lj2[j][i] = lj2[i][j];
  lj3[j][i] = lj3[i][j];
  lj4[j][i] = lj4[i][j];
  offset[j][i] = offset[i][j];

  // check interior rRESPA cutoff

  if (cut_respa && cut[i][j] < cut_respa[3])
    error->all(FLERR,"Pair cutoff < Respa interior cutoff");

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);

    double sig2 = sigma[i][j]*sigma[i][j];
    double sig6 = sig2*sig2*sig2;
    double rc3 = cut[i][j]*cut[i][j]*cut[i][j];
    double rc6 = rc3*rc3;
    double rc9 = rc3*rc6;
    etail_ij = 8.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (sig6 - 3.0*rc6) / (9.0*rc9);
    ptail_ij = 16.0*MY_PI*all[0]*all[1]*epsilon[i][j] *
      sig6 * (2.0*sig6 - 3.0*rc6) / (9.0*rc9);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNN::write_restart(FILE *fp) {
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&epsilon[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNN::read_restart(FILE *fp) {
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&epsilon[i][j],sizeof(double),1,fp);
          fread(&sigma[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&epsilon[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairNN::write_restart_settings(FILE *fp) {
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairNN::read_restart_settings(FILE *fp) {
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ----------------------------------------------------------------------
   proc 0 writes to data file
------------------------------------------------------------------------- */

void PairNN::write_data(FILE *fp) {
  for (int i = 1; i <= atom->ntypes; i++)
    fprintf(fp,"%d %g %g\n",i,epsilon[i][i],sigma[i][i]);
}

/* ----------------------------------------------------------------------
   proc 0 writes all pairs to data file
------------------------------------------------------------------------- */

void PairNN::write_data_all(FILE *fp) {
  for (int i = 1; i <= atom->ntypes; i++)
    for (int j = i; j <= atom->ntypes; j++)
      fprintf(fp,"%d %d %g %g %g\n",i,j,epsilon[i][j],sigma[i][j],cut[i][j]);
}

/* ---------------------------------------------------------------------- */

double PairNN::single(int i, 
                      int j, 
                      int itype, 
                      int jtype, 
                      double rsq,
                      double factor_coul, 
                      double factor_lj,
                      double &fforce) {

  double r2inv,r6inv,forcelj,philj;
  r2inv = 1.0/rsq;
  r6inv = r2inv*r2inv*r2inv;
  forcelj = r6inv * (lj1[itype][jtype]*r6inv - lj2[itype][jtype]);
  fforce = factor_lj*forcelj*r2inv;

  philj = r6inv*(lj3[itype][jtype]*r6inv-lj4[itype][jtype]) -
    offset[itype][jtype];
  return factor_lj*philj;
}

/* ---------------------------------------------------------------------- */

void *PairNN::extract(const char *str, int &dim) {
  dim = 2;
  if (strcmp(str,"epsilon") == 0) return (void *) epsilon;
  if (strcmp(str,"sigma") == 0) return (void *) sigma;
  return NULL;
}

double PairNN::network(double x) {
    arma::mat in = arma::zeros<arma::mat>(1,1);
    in(0,0) = x;
    m_preActivations[0] = in;
    m_activations   [0] = m_preActivations[0];

    // hidden layers
    for (int i=0; i < m_layers; i++) {
        // weights and biases starts at first hidden layer:
        // weights[0] are the weights connecting inputGraph layer to first hidden layer
        m_preActivations[i+1] = m_activations[i]*m_weights[i] + m_biases[i];
        m_activations   [i+1] = sigmoid(m_preActivations[i+1]);
    }

    // linear activation for output layer
    m_preActivations[m_layers+1] = m_activations[m_layers]*m_weights[m_layers] + m_biases[m_layers];
    m_activations   [m_layers+1] = m_preActivations[m_layers+1];

    // return activation of output neuron
    return m_activations[m_layers+1](0,0);
}


arma::mat PairNN::sigmoid(arma::mat matrix) {

  return 1.0/(1 + arma::exp(-matrix));
}

arma::mat PairNN::sigmoidDerivative(arma::mat matrix) {

  arma::mat sigmoidMatrix = sigmoid(matrix);
  return sigmoidMatrix % (1 - sigmoidMatrix);
}


double PairNN::backPropagation() {
  // find derivate of output w.r.t. intput, i.e. dE/dr_ij
  // need to find the "error" terms for all the nodes in all the layers

  // the derivative of the output neuron's activation function w.r.t.
  // its inputGraph is propagated backwards.
  // the output activation function is f(x) = x, so this is 1
  arma::mat output(1,1); output.fill(1);
  m_derivatives[m_layers+1] = output;

  // we can thus compute the error vectors for the other layers
  for (int i=m_layers; i > 0; i--) {
      m_derivatives[i] = ( m_derivatives[i+1]*m_weightsTransposed[i] ) % sigmoidDerivative(m_preActivations[i]);
  }

  // linear activation function for inputGraph neurons
  m_derivatives[0] = m_derivatives[1]*m_weightsTransposed[0];

  return m_derivatives[0](0,0);
}


void PairNN::load(char* fileName) {
    ifstream inFile;
    //inFile.open(fileName, std::ios::in);
    //inFile.open("/Users/morten/Documents/Master/TFPotential/Visualization/ljnet/network-80000", std::ios::in);
    inFile.open("/Users/morten/Documents/Master/TFPotential/Visualization/network-935000_copy.txt", std::ios::in);
    bool open = false;
    if (inFile.is_open()) {
        if (inFile.good()) {
            cout << "File " << fileName << " successfully opened." << endl;
            open = true;
        }
    }
    if (! open) {
        cout << "Could not open file " << fileName << " successfully opened." << endl;
        cout << "Exiting." << endl;
        exit(1);
    }

    // First line:  inputs   layers   nodes   outputs
    inFile >> m_inputs >> m_layers >> m_nodes >> m_outputs;

    if (m_inputs > 1 || m_outputs > 1) {
        cout << "Multi input/output network reading not implemented yet." << endl;
        cout << "Exiting." << endl;
        exit(1);
    }

    m_preActivations.resize(m_layers+2);
    m_activations.resize(m_layers+2);
    m_derivatives.resize(m_layers+2);

    m_weights.resize(m_layers+1);
    m_biases.resize(m_layers+1);

    // Read first layer weights
    m_weights.at(0) = zeros<arma::mat>(1, m_nodes);
    for (int i=0; i < m_nodes; i++) {
        inFile >> m_weights.at(0)(0,i);
    }

    m_biases.at(0) = zeros<arma::mat>(1, m_nodes);
    // Read the first layer biases
    for (int i=0; i < m_nodes; i++) {
        inFile >> m_biases.at(0)(0,i);
    }

    // If multiple layers, the middle weights have dimensions (nodes x nodes)
    if (m_layers > 1) {
        // The (nodes x nodes) size continues until last layer _before_ output
        for (int k=1; k < m_layers; k++) {
            m_weights.at(k) = zeros<arma::mat>(m_nodes, m_nodes);

            // We traverse the file in a column-major fashion when reading line
            // by line
            for (int i=0; i < m_nodes; i++) {
                for (int j=0; j < m_nodes; j++) {
                    inFile >> m_weights.at(k)(i,j);
                }
            }

            // After the weight, comes the corresponding bias vector
            m_biases.at(k) = zeros<arma::mat>(1,m_nodes);
            for (int i=0; i < m_nodes; i++) {
                inFile >> m_biases.at(k)(0,i);
            }
        }
    }

    // Last layer weights have the shape (nodes x outputs)
    m_weights.at(m_layers) = zeros<arma::mat>(m_nodes, 1);
    for (int i=0; i < m_nodes; i++) {
        inFile >> m_weights.at(m_layers)(i,0);
    }
    // Last bias
    m_biases.at(m_layers) = zeros<arma::mat>(1, 1);
    inFile >> m_biases.at(m_layers)(0,0);
    inFile.close();

    m_weightsTransposed.resize(m_layers+1);
    for (int i=0; i < m_weights.size(); i++) {
        m_weightsTransposed[i] = m_weights[i].t();
    }
}
