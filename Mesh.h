#include <iostream>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <petsc.h>

class Mesh {
	public:
    /** resX, resY, lengthX, lengthY */
	Mesh( DM*, int, int, double, double );
	~Mesh();
	void Print();
	/** get mesh dimensionality */
  int GetDim(int*);
  int GetNodeCount(int*);
	/** get number of nodes per el */
  int GetNodeCountPerEl(int*);
  int GetLocalElementSize();

	/** get number of nodes in given dimension */
  int GetNodeRes(const int, int *);

	/** get node seperation */
	int GetNodeSep( double * );

	/** get number of elements in mesh */
  int GetElementCount(int*);

	/** return the nodes in an element **/
  int GetElementNodes(int elID, int *list);

  int GetGaussPointCount(int*);
  /** get gauss point coords, for the part_I-th gauss point */
  void GaussPointCoord( const int part_I, double** );
  void EvaluateShapeFunc( const double *pos, double *Ni );
  /** dNi_dx: local shape function derivs */
  void Evaluate_dNxFunc(  const double *pos, double dNi_dx[][4] );
  void Evaluate_GNxFunc( const double *pos, const int elID, double GNx[][4], double *detJac );
  bool NodeCoords( const int gNodeID, double *pos );
	/** ijk mapping to global node id */
  int NodeMap_ijk2g( const int *ijk, int *gID );

	private:
	int gNodeCountXYZ[3];
  double lengthXYZ[3];
	int gElCountXYZ[3];
	int elCount;
	int nodesPerEl;
	int elementNodeCount;
	int dim;
	double** gCoords;
	int** nbrNodeArray;
	int** elNodeArray;
	int** nodeElArray;
  DM* da;
  int local_elements;
};
