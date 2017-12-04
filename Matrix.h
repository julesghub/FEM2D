#include<iostream>
#include<string.h>
//#include<petsc.h>

class Matrix {
	public:
	Matrix(Mesh*, Field*, const char*);
	~Matrix();;
	void Print();
	void assembleLocalMassMatrix( int el_ID, double *M);
	void assembleLaplacian( int el_ID, double *M);
	void assembleLumpedMassMatrix( int el_ID, double *M);
	void assembleM_K( int el_ID, double factor, double *M);
	void GlobalAssembleMassMatrix();
//	Vec myVector;

	private:
	Field  *field;
  Mesh   *mesh;
  int    dof[3];
	char   *name;
};
