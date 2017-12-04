#include "Mesh.h"
#include "Field.h"
#include "Matrix.h"

Matrix::Matrix(Mesh *inputMesh, Field *inputField, const char inputString[]) {
  printf("Building Matrix\n");
  int dofPerNode, nodeRes[3];

  field = inputField;
  mesh = inputMesh;
  name = new char [strlen(inputString)+1];
  strcpy( name, inputString );

  field->GetDofCount( &dofPerNode );

  mesh->GetNodeRes(0, &nodeRes[0] );
  mesh->GetNodeRes(1, &nodeRes[1] );
  mesh->GetNodeRes(2, &nodeRes[2] );

  dof[0] = nodeRes[0]*dofPerNode;
  dof[1] = nodeRes[1]*dofPerNode;
  dof[2] = nodeRes[2]*dofPerNode;

}

Matrix::~Matrix() {
  printf("Deleting Matrix\n");
  delete [] name;
}

void Matrix::Print() {
}

void Matrix::GlobalAssembleMassMatrix() {
  int el_I, elCount;

  mesh->GetElementCount( &elCount );
  for( el_I = 0 ; el_I < elCount ; el_I++ ) {

  }
}
void Matrix::assembleLocalMassMatrix( int el_ID, double *M ) {
  int nodesPerEl = 0;
  int gPointPerEl = 0;
  double *coord;
  double Ni[4], GNx[2][4], detJac;
  double A[4][4];

  mesh->GetNodeCountPerEl( &nodesPerEl );
  mesh->GetGaussPointCount( &gPointPerEl );

  int i,j,part_I;

  /** initialise A */
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      A[i][j] = 0;
    }
  }

  for( part_I = 0 ; part_I < gPointPerEl ; part_I++ ) {
    mesh->GaussPointCoord( part_I, &coord );
    mesh->EvaluateShapeFunc( coord, Ni );
    mesh->Evaluate_GNxFunc( coord, el_ID, GNx, &detJac );
    for( i = 0 ; i < nodesPerEl ; i++ ) {
      for( j = 0 ; j < nodesPerEl ; j++ ) {
        A[i][j] += Ni[i]*Ni[j]*detJac;
      }
    }
  }

  /** map matrix into single array */
  int add = 0;
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j <nodesPerEl ; j++ ) {
      M[add] = A[i][j];
      add++;
    }
  }
}

void Matrix::assembleLaplacian( int el_ID, double *M ) {
  int nodesPerEl = 0;
  int gPointPerEl = 0;
  int dim = 0;
  double *coord;
  double GNx[2][4], detJac;
  double A[4][4];

  mesh->GetNodeCountPerEl( &nodesPerEl );
  mesh->GetDim( &dim );
  mesh->GetGaussPointCount( &gPointPerEl );

  int i,j,part_I;

  /** initialise A */
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      A[i][j] = 0;
    }
  }

  for( part_I = 0 ; part_I < gPointPerEl ; part_I++ ) {
    mesh->GaussPointCoord( part_I, &coord );
    mesh->Evaluate_GNxFunc( coord, el_ID, GNx, &detJac );
    for( i = 0 ; i < nodesPerEl ; i++ ) {
      for( j = 0 ; j < nodesPerEl ; j++ ) {
        for( int dim_I = 0 ; dim_I < dim ; dim_I++ ) {
          A[i][j] -= detJac*GNx[dim_I][i]*GNx[dim_I][j];
        }
      }
    }
  }

  /** map matrix into single array */
  int add = 0;
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      M[add] = A[i][j];
      add++;
    }
  }
}

void Matrix::assembleLumpedMassMatrix( int el_ID, double *M ) {
  int nodesPerEl = 0;
  int gPointPerEl = 0;
  double *coord;
  double Ni[4], GNx[2][4], detJac;
  double A[4][4];

  mesh->GetNodeCountPerEl( &nodesPerEl );
  mesh->GetGaussPointCount( &gPointPerEl );

  int i,j,part_I;

  /** initialise A */
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    M[i] = 0;
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      A[i][j] = 0;
    }
  }

  for( part_I = 0 ; part_I < gPointPerEl ; part_I++ ) {
    mesh->GaussPointCoord( part_I, &coord );
    mesh->EvaluateShapeFunc( coord, Ni );
    mesh->Evaluate_GNxFunc( coord, el_ID, GNx, &detJac );
    for( i = 0 ; i < nodesPerEl ; i++ ) {
      for( j = 0 ; j < nodesPerEl ; j++ ) {
        A[i][j] += Ni[i]*Ni[j]*detJac;
      }
    }
  }

  /** map matrix into single array */
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j <nodesPerEl ; j++ ) {
      M[i] += A[i][j];
    }
  }
}

void Matrix::assembleM_K( int el_ID, double factor, double *M ) {
  int nodesPerEl = 0;
  int gPointPerEl = 0;
  int dim = 0;
  double *coord;
  double GNx[2][4], Ni[4], detJac;
  double A[4][4];

  mesh->GetNodeCountPerEl( &nodesPerEl );
  mesh->GetDim( &dim );
  mesh->GetGaussPointCount( &gPointPerEl );

  int i,j,part_I;

  /** initialise A */
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      A[i][j] = 0;
    }
  }

  for( part_I = 0 ; part_I < gPointPerEl ; part_I++ ) {
    mesh->GaussPointCoord( part_I, &coord );
    mesh->EvaluateShapeFunc( coord, Ni );
    mesh->Evaluate_GNxFunc( coord, el_ID, GNx, &detJac );
    for( i = 0 ; i < nodesPerEl ; i++ ) {
      for( j = 0 ; j < nodesPerEl ; j++ ) {
        A[i][j] += detJac * ( Ni[i]*Ni[j] + factor * (GNx[0][i]*GNx[0][j]+GNx[1][i]*GNx[1][j] ) );
      }
    }
  }

  /** map matrix into single array */
  int add = 0;
  for( i = 0 ; i < nodesPerEl ; i++ ) {
    for( j = 0 ; j < nodesPerEl ; j++ ) {
      M[add] = A[i][j];
      add++;
    }
  }
}
