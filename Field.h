#include<iostream>
#include<cassert>
#include <petsc.h>

class Field {
	public:
	Field(Mesh*, int);
	~Field();
	void Print();
	void gnuplot( const char *filename );
	void GetDofCount(int *);
	void GetBCInfo(int *numBCs, int **bcList, double **bcValues);
	void SetBCDof();
	void MapVec2Dof( Vec *myVec );
	/* map dirichlet bc into vector */
	void MapDirichletIntoVec( Vec *vec );
	void SetBCDof_OnWall( int planeAxis, int minMax, const double wallBC );
  void SetBCInfo(int numBcs, int* bcList, double *bcValues);

	private:
  int    *_bcList;
	int    _numBCs;
  Mesh   *mesh;
	int    _dofCount;
	int    _num_unkn;
  /* my own version of unknowns, should just be petsc Vec */
	double *_dof;
	double *_bcValues;

};
