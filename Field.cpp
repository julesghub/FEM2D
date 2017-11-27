
#include "Mesh.h"
#include "Field.h"


Field::Field(Mesh* inputMesh, int inputDofCount) {
  int nodeRes[3];
  printf("Building Field\n");
  _dofCount = inputDofCount;
  mesh = inputMesh;

  mesh->GetNodeRes( 0, &nodeRes[0] );
  mesh->GetNodeRes( 1, &nodeRes[1] );

  /* overestimate because of BCs ... oh well */
  _num_unkn = (nodeRes[0])*(nodeRes[1])*_dofCount;
  /* build all field memory */
  _dof = new double[ _num_unkn ];
  memset( _dof, 0, sizeof(double)*_num_unkn );

  /* build full _bcValues and _bcList (we'll only index as much as _numBCs though */
  _bcValues = new double [2*nodeRes[0] + 2*nodeRes[1]];
  _bcList = new int [2*nodeRes[0] + 2*nodeRes[1]];
  _numBCs = 0;
}

Field::~Field() {
  printf("Deleting Field\n");
  delete [] _dof;
  delete [] _bcValues;
  delete [] _bcList;
}

void Field::Print() {
  printf("Field looks like:\n");
  for( int ii = 0 ; ii < _num_unkn ; ii++ ) {
    printf("%lf\n", _dof[ii] );
  }
}

/** Get dof count on field */
void Field::GetDofCount( int *count) {
  *count = this->_dofCount;
}

void Field::SetBCDof() {
  int ii;
  for( ii = 0 ; ii < _numBCs; ii++ ) {
    _dof[ _bcList[ii] ] = _bcValues[ii];
  }
}

void Field::MapDirichletIntoVec( Vec *vec ) {
  VecSetValues( *vec, _numBCs, _bcList, _bcValues, INSERT_VALUES );
}

void Field::SetBCDof_OnWall( int planeAxis, int minMax, const double wallBC ) {
  /*
  Writes dirichlet BC info to the Field c information only - not the Petsc Vec

  Params
  ------
  planeAxis : 0 is x-axis, 1 is y-axis
  minMax    : 0 is min indicies, 1 is max indicies
  wallBC    : value to impose on those nodes.
  */

  int res_A, dim, ii, ijk[3];
  int maxParam, gId;

  /** ensure minMax is either the min parametrisation or the max */
  mesh->GetNodeRes( planeAxis, &maxParam );
  assert( minMax == 0 || minMax == maxParam-1 );

  mesh->GetDim(&dim);

  assert( dim != 3 );

  if (planeAxis == 1 ) {
    // if planeAxis is in Y then get number of X nodes
    mesh->GetNodeRes( 0, &res_A );
    ijk[1] = minMax;
    // loop in x-axis
    for( ii = 0 ; ii < res_A ; ii++ ) {
      ijk[0] = ii;
      mesh->NodeMap_ijk2g( ijk, &gId );
      _dof[gId] = wallBC;
      // set bc lists
      _bcList[_numBCs] = gId;
      _bcValues[_numBCs] = wallBC;
      _numBCs++;
    }

  } else {
    // else assume plane is in X and get number of Y nodes
    mesh->GetNodeRes( 1, &res_A );
    ijk[0] = minMax;
    // loop in y-axis
    for( ii = 0 ; ii < res_A ; ii++ ) {
      ijk[1] = ii;
      mesh->NodeMap_ijk2g( ijk, &gId );
      _dof[gId] = wallBC;
      // set bc lists
      _bcList[_numBCs] = gId;
      _bcValues[_numBCs] = wallBC;
      _numBCs++;
    }
  }
}

void Field::GetBCInfo(int *numBCs, int **bcList, double **bcValues) {
  *numBCs = _numBCs;
  *bcList = _bcList;
  *bcValues = _bcValues;
}

void Field::MapVec2Dof( Vec *myVec ) {
  int ii;
  int unknInd[1000];

  /* need this array for petsc indices */
  for( ii = 0 ; ii < _num_unkn ; ii++ )
    unknInd[ii] = ii;

  /* get values from myVec and put them in dof */
  VecGetValues( *myVec, _num_unkn, unknInd, _dof );

  /* overwrite BC values in dof */
  SetBCDof();
}

void Field::gnuplot( const char *filename) {
  int num_node, node_i;
  double coord[3] = {0,0,0};
  FILE *filePtr = NULL;

  filePtr = fopen( filename, "w" );
  if( filePtr == NULL ) {
    printf( "\n\nError opening %s, in %s\n\n", filename, __FILE__ ); assert(0);
  }

  mesh->GetNodeCount( &num_node );

  for( node_i = 0 ; node_i < num_node ; node_i++ ) {
    mesh->NodeCoords( node_i, coord );
    // fprintf( filePtr, "%g %g %g %g\n", coord[0], coord[1], coord[2], _dof[node_i] );
    fprintf( filePtr, "%g %g %g\n", coord[0], coord[1], _dof[node_i] );

  }

  fclose( filePtr );
}
