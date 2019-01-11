/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(nn,PairNN)

#else

#ifndef LMP_PAIR_NN_H
#define LMP_PAIR_NN_H

#include "pair.h"
#include <armadillo>
#include <vector>

namespace LAMMPS_NS {

class PairNN : public Pair {
 public:
  PairNN(class LAMMPS *);
  virtual ~PairNN();
  virtual void compute(int, int);
  void settings(int, char **);
  void coeff(int, char **);
  void init_style();
  void init_list(int, class NeighList *);
  double init_one(int, int);
  void write_restart(FILE *);
  void read_restart(FILE *);
  void write_restart_settings(FILE *);
  void read_restart_settings(FILE *);
  void write_data(FILE *);
  void write_data_all(FILE *);
  double single(int, int, int, int, double, double, double, double &);
  void *extract(const char *, int &);

  double network(double x);
  double backPropagation();
  void load(char* fileName);

 protected:
  double cut_global;
  double **cut;
  double **epsilon,**sigma;
  double **lj1,**lj2,**lj3,**lj4,**offset;
  double *cut_respa;
  std::ofstream outFile;
  int m_layers;
  int m_nodes;
  int m_inputs;
  int m_outputs;

  std::vector<arma::mat> m_weights            = std::vector<arma::mat>();
  std::vector<arma::mat> m_weightsTransposed  = std::vector<arma::mat>();
  std::vector<arma::mat> m_biases             = std::vector<arma::mat>();
  std::vector<arma::mat> m_preActivations     = std::vector<arma::mat>();
  std::vector<arma::mat> m_activations        = std::vector<arma::mat>();
  std::vector<arma::mat> m_derivatives        = std::vector<arma::mat>();
  arma::mat sigmoid(arma::mat matrix);
  arma::mat sigmoidDerivative(arma::mat matrix);

  virtual void allocate();
};

}

#endif
#endif

/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Incorrect args for pair coefficients

Self-explanatory.  Check the input script or data file.

E: Pair cutoff < Respa interior cutoff

One or more pairwise cutoffs are too short to use with the specified
rRESPA cutoffs.

*/
