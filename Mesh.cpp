#include"Mesh.h"

Mesh::Mesh( DM* input_da, int elResX, int elResY, double xLength, double yLength ) {
  int el_I;
  PetscInt lElNum, lNodeNum;
  const PetscInt* e_n_graph;

  da = input_da;

  DMDASetUniformCoordinates(da[0],0.0,xLength,0.0,yLength,0.0,1.0); // unused dimension isn't used

  DMDAGetElements( da[0], &lElNum, &lNodeNum, &e_n_graph );
  elCount = (int)lElNum;

  printf("Building mesh\n");
  lengthXYZ[0] = xLength;
  lengthXYZ[1] = yLength;
  /* initialise mesh: simple quads to tets */
  nodesPerEl = 4;
  gElCountXYZ[0] = elResX;
  gElCountXYZ[1] = elResY;
  gElCountXYZ[2] = 0;
  gNodeCountXYZ[0] = gElCountXYZ[0]+1;
  gNodeCountXYZ[1] = gElCountXYZ[1]+1;
  gNodeCountXYZ[2] = 0;

  gElCount = gElCountXYZ[0] * gElCountXYZ[1];
  dim = 2;
  if( gElCountXYZ[2] != 0 ) {
    gElCount *= gElCountXYZ[2];
    dim = 3;
  }

  elNodeArray = new int* [elCount];
  for( el_I = 0 ; el_I < elCount ; el_I++ ) {
    elNodeArray[el_I] = new int[lNodeNum];
    memcpy( elNodeArray[el_I], &e_n_graph[lNodeNum*el_I], lNodeNum*sizeof(int) );
  }

  /* build element node table; ordering
   * 3 ------ 2
   * |        |
   * |        |
   * 0 ------ 1
   */

  /* build gaussPoints */
  int g_i;
  gCoords = new double* [4];
  for( g_i = 0 ; g_i < 4 ; g_i++ ) {
    gCoords[g_i] = new double[3];
    memset( gCoords[g_i], 0, 3*sizeof(double) );
  }

  double oneOnSqrt3 = 1/sqrt(3);
  gCoords[0][0] = -1*oneOnSqrt3; gCoords[0][1] = -1*oneOnSqrt3;
  gCoords[1][0] =  1*oneOnSqrt3; gCoords[1][1] = -1*oneOnSqrt3;
  gCoords[2][0] =  1*oneOnSqrt3; gCoords[2][1] =  1*oneOnSqrt3;
  gCoords[3][0] = -1*oneOnSqrt3; gCoords[3][1] =  1*oneOnSqrt3;

  DMDARestoreElements(da[0], &lElNum, &lNodeNum, &e_n_graph);

}

Mesh::~Mesh() {
  printf("Deleting mesh\n");
  int el_I;
  for( el_I = 0 ; el_I < elCount ; el_I++ )
    delete [] elNodeArray[el_I];
  delete [] elNodeArray;

  int g_i;
  for( g_i = 0 ; g_i < 4 ; g_i++ )  {
    delete [] gCoords[g_i];
  }
  delete [] gCoords;

}

void Mesh::Print() {
  printf("\n*** Mesh::Print()\n");
  printf("The gElCountXYZ = %d %d %d ...elCount %d\n", gElCountXYZ[0], gElCountXYZ[1], gElCountXYZ[2], elCount );
  printf("The gNodeCountXYZ = %d %d %d ...nodeCount %d \n", gNodeCountXYZ[0], gNodeCountXYZ[1], gNodeCountXYZ[2],
                                                           gNodeCountXYZ[0] * gNodeCountXYZ[1] );
  printf("elNodeArray:\n");
  int el_I;
  for( el_I = 0 ; el_I < elCount ; el_I++ ) {
    printf(" El %d = { %d %d %d %d }\n", el_I, elNodeArray[el_I][0], elNodeArray[el_I][1], elNodeArray[el_I][2], elNodeArray[el_I][3] );
  }
  printf(" Printing local gaussPoint coords:\n");
  for( int g_i = 0 ; g_i < 4; g_i++ ) {
    printf("(%g %g)\n", gCoords[g_i][0], gCoords[g_i][1] );
  }

  printf("\n*** Finished Mesh::Print()\n");
}

int Mesh::GetGaussPointCount(int* count) {
  *count = 4; /* IT's HARD CODED */
  return 0;
}

int Mesh::GetDim(int* count) {
  *count = dim;
  return 0;
}

int Mesh::GetElementCount(int* count) {
  *count = elCount;
  return 0;
}

int Mesh::GetNodeSep( double *sep ) {
  assert( gNodeCountXYZ[0] );
  assert( gNodeCountXYZ[1] );
  sep[0] = lengthXYZ[0]/gElCountXYZ[0];
  sep[1] = lengthXYZ[0]/gElCountXYZ[1];
  if( dim == 3 ) {
    assert( lengthXYZ[2] ); sep[2] = gNodeCountXYZ[2]/lengthXYZ[2];
  }
  return 0;
}
int Mesh::GetNodeCount(int* count) {
  *count = gNodeCountXYZ[0]*gNodeCountXYZ[1];

  /** if 3d **/
  if(dim==3)
    *count = *count * gNodeCountXYZ[2];

  return 0;
}
int Mesh::GetNodeCountPerEl(int* count) {
  *count = nodesPerEl;
  return 0;
}

/** return the number of nodes in a give dir */
int Mesh::GetNodeRes(const int axis, int *count) {
  *count = gNodeCountXYZ[axis];
  return 0;
}
void Mesh::GaussPointCoord( const int part_I, double **coord ) {
  *coord = gCoords[part_I];
}

int Mesh::GetLocalElementSize() {
  return elCount;
}

int Mesh::GetElementNodes(int elID, int *list) {
  // const PetscInt *e_n_graph;
  // PetscInt lNodeNum, lElNum;
  // DMDAGetElements( da[0], &lElNum, &lNodeNum, &e_n_graph );
  // memcpy( list, &e_n_graph[lNodeNum*elID], sizeof(int)*lNodeNum );
  // DMDARestoreElements( da[0], &lElNum, &lNodeNum, &e_n_graph );
  
  memcpy( list, elNodeArray[elID], 4*sizeof(int) );
  return 0;
}

void Mesh::EvaluateShapeFunc( const double *pos, double *Ni ) {
  double xi, eta;

  xi = pos[0];
  eta = pos[1];

  Ni[0] = 0.25 * (1-xi) * (1-eta);
  Ni[1] = 0.25 * (1+xi) * (1-eta);
  Ni[2] = 0.25 * (1+xi) * (1+eta);
  Ni[3] = 0.25 * (1-xi) * (1+eta);
}

void Mesh::Evaluate_dNxFunc( const double *pos, double dNi_dx[][4] ) {
  double xi, eta;

  xi = pos[0];
  eta = pos[1];
  /** derivs with repect to xi */
  dNi_dx[0][0] = -0.25 * (1-eta);
  dNi_dx[0][1] =  0.25 * (1-eta);
  dNi_dx[0][2] =  0.25 * (1+eta);
  dNi_dx[0][3] = -0.25 * (1+eta);

  /** derivs with repect to eta */
  dNi_dx[1][0] = -0.25 * (1-xi);
  dNi_dx[1][1] = -0.25 * (1+xi);
  dNi_dx[1][2] =  0.25 * (1+xi);
  dNi_dx[1][3] =  0.25 * (1-xi);
}

void Mesh::Evaluate_GNxFunc( const double *pos, const int elID, double GNx[][4], double *detJac ){
  double jac[3][3], D;
  double GNi[2][4], nodeCoord[3];
  int node_I;

  /** initialise jac */
  jac[0][0] = jac[0][1] = jac[0][2] = 0;
  jac[1][0] = jac[1][1] = jac[1][2] = 0;
  jac[2][0] = jac[2][1] = jac[2][2] = 0;

  /** get the local derivative of this position */
  Evaluate_dNxFunc( pos, GNi );
  
  
  const PetscInt *e_n_graph;
  PetscInt lNodeNum, lElNum;
  DMDAGetElements( da[0], &lElNum, &lNodeNum, &e_n_graph );
  ISLocalToGlobalMapping ltogm;
  const PetscInt *array;
  DMGetLocalToGlobalMapping(da[0], &ltogm);
  ISLocalToGlobalMappingGetIndices(ltogm,&array);
  
  for( node_I = 0 ; node_I < nodesPerEl ; node_I++ ) {
    // get the global id
    NodeCoords( e_n_graph[4*elID+node_I],  nodeCoord);
    jac[0][0] += ( GNi[0][node_I]*nodeCoord[0] );
    jac[0][1] += ( GNi[0][node_I]*nodeCoord[1] );

    jac[1][0] += ( GNi[1][node_I]*nodeCoord[0] );
    jac[1][1] += ( GNi[1][node_I]*nodeCoord[1] );
  }
  DMDARestoreElements( da[0], &lElNum, &lNodeNum, &e_n_graph );
  /** calculate detJac */
  D = jac[0][0]*jac[1][1] - jac[0][1]*jac[1][0];
  *detJac = D;

  double tmp;
  /* invert the jacobian matrix A^-1 = adj(A)/det(A) */
  tmp = jac[0][0];
  jac[0][0] = jac[1][1]/D;
  jac[1][1] = tmp/D;
  jac[0][1] = -jac[0][1]/D;
  jac[1][0] = -jac[1][0]/D;

  /* get apply local derivs and jac */
  int dx, n, dxi;
  double globalSF_DerivVal;
  for( dx=0; dx<dim; dx++ ) {
    for( n=0; n<nodesPerEl; n++ ) {

      globalSF_DerivVal = 0.0;
      for(dxi=0; dxi<dim; dxi++) {
        globalSF_DerivVal = globalSF_DerivVal + GNi[dxi][n] * jac[dx][dxi];
      }

      GNx[dx][n] = globalSF_DerivVal;
    }
  }
}

bool Mesh::NodeCoords( const int ndoeId, double *pos ) {
  PetscInt dim;
  Vec nodeCoords;

  GetDim( &dim );
  
  // Only has "local" nodeCoords 
  DMGetCoordinatesLocal(*da, &nodeCoords);

  PetscScalar *coords=NULL;
  VecGetArray(nodeCoords, &coords);

  memcpy( pos, &coords[dim*ndoeId], sizeof(double)*dim );
  VecRestoreArray(nodeCoords, &coords);
  return true;

}

int Mesh::NodeMap_ijk2g( const int *ijk, int *gID ) {
  assert( ijk[0] <= (gNodeCountXYZ[0]-1) );
  assert( ijk[1] <= (gNodeCountXYZ[1]-1) );

  *gID = 0;
  *gID = ijk[0] + ijk[1]*gNodeCountXYZ[0];

  if(dim == 3) {
    assert(0);
  }

  return 0;
}
