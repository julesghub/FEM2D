#include<iostream>

#include <petsc.h>
#include <petscdm.h>
#include "petscsys.h"   
#include "petscviewerhdf5.h"


#include "Mesh.h"
#include "Field.h"
#include "Matrix.h"

static char help[] = "Solve 2D laplace equation, constant coefficient. T(y_min)=0.1, T(y_max)=1;\n \
   OPTIONS:\n \
   -max_x\n \
   -max_y\n \
   -dt_fac\n \
   -solver_type\n \
 \
 Grid size and many other options can be change via petsc options\n\n \
 \
 To visualise results open up gnuplot and use:\n \
    `splot 'field.out' u 1:2:3`\n";

void SolveEuler_FE_LumpedMM(
    Field *field,
    Mat *K, Mat *M,
     Vec *lumped_mm, Vec *phi,
    int num_timeSteps, double dt );

void SolveForwardEuler(
    Field* field,
    Mat *K, Mat *M,
    Vec *lumped_mm, Vec *phi,
    int num_timeSteps, double dt );

void SolvePredictorCorrector(Field* field,
   Mat *K, Mat *M,
   Vec *lumped_mm, Vec *phi,
   int num_timeSteps, double dt );

void SolveBackwardEuler(
    Field* field,
    Mat *K, Mat *M,
    Vec *lumped_mm, Vec *phi,
    int num_timeSteps, double dt );

void SolveCN(
    Field* field,
    Mat *K, Mat *M,
    Vec *phi,
    int num_timeSteps, double dt );

double max_double( double a, double b ){
  return (a > b) ? a : b; 
}

double min_double( double a, double b ){
  return (a < b) ? a : b; 
}

int main(int argc, char **argv) {

  DM da;
  Mat M, K;
  Vec phi, lumped_mm;

  double max[] = {1,1};
  double dt, minSep, dt_fac;
  double tmpMem[16];
  int    num_timeSteps, solver_type;
  int    rank, nProcs;
  int    ijk_sizes[3], n_i;
  
  PetscBool flg;
  PetscLogDouble petscMem;
  PetscErrorCode ierr;
  
  // start up petsc
  ierr = PetscInitialize( &argc, &argv, (char*)0, help ); if( ierr ) return ierr;
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&nProcs);
  
  // get cmd args 
  PetscOptionsGetInt(PETSC_NULL, 0, "-n", &num_timeSteps, &flg);
  if( !flg ) num_timeSteps = 1; 
  PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-max_x", &max[0], &flg);
  PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-max_y", &max[1], &flg);
  PetscOptionsGetScalar(PETSC_NULL, 0, "-dt_fac", &dt_fac, &flg);
  if( !flg ) dt_fac = 0.5;
  PetscOptionsGetInt(PETSC_NULL,0,"-solver_type",&solver_type,&flg);
  if( !flg ) solver_type=0;
  // create dm
  DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, /* boundary types in x/y */
               DMDA_STENCIL_BOX,                  /* neighbour shape */
               10+1, 10+1,                        /* grid size */
               PETSC_DECIDE,PETSC_DECIDE,         /* nprocs in each direction */
               1,1,                               /* dof and width */
               NULL, NULL, &da);                  /* nodes along x & y */
  DMSetFromOptions(da);
  DMSetUp(da); // MUST CALL!
  DMDAGetInfo(da,0,&ijk_sizes[0],&ijk_sizes[1],&ijk_sizes[2],0,0,0,0,0,0,0,0,0);
  int _lElNum, lElNum, _nodesPerEl;
  const int *_e_n_graph;
  DMDAGetElements( da, &_lElNum, &_nodesPerEl, &_e_n_graph );
  lElNum = _lElNum;
  DMDARestoreElements(da, &_lElNum, &_nodesPerEl, &_e_n_graph);
  
  Mesh* mesh = new Mesh( &da, ijk_sizes[0], ijk_sizes[1], max[0], max[1] );
  Field* field   = new Field(mesh, 1);
  Matrix* matrix = new Matrix(mesh, field, "matrix1");

  // PetscSynchronizedPrintf(PETSC_COMM_WORLD, "I'm %d: I have %d local elements\n", rank, lElNum);
  // PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);

  // calculate timesetp
  minSep = min_double( (max[0]/ijk_sizes[0]),  (max[1]/ijk_sizes[1]) );
  dt = dt_fac*(minSep*minSep);

  PetscMallocGetCurrentUsage( &petscMem );
  PetscPrintf(PETSC_COMM_WORLD, "*****\nPetscMalloc, before run %f\n*******\n", petscMem );
  
  // Make the Matrices
  DMCreateMatrix(da, &M);
  DMCreateMatrix(da, &K);
  MatSetFromOptions(M); MatSetUp(M);
  MatSetFromOptions(K); MatSetUp(K);
  
  /** 2) assemble a phi */
  MatCreateVecs(M,&phi, NULL);
  PetscObjectSetName((PetscObject)phi, "field");
  VecSetFromOptions( phi );
  VecScale( phi, 0 );
  VecDuplicate( phi, &lumped_mm );  
  /* assemble a laplacian matrix */
  PetscInt nodeList[4];
    
  /* assemble across elements */
  for( int el_i = 0 ; el_i < lElNum ; el_i++ ) {
    mesh->GetElementNodes(el_i, nodeList);

    matrix->assembleLocalMassMatrix( el_i, tmpMem );
    MatSetValuesLocal(M, 4, nodeList, 4, nodeList, tmpMem, ADD_VALUES );
    
    matrix->assembleLaplacian( el_i, tmpMem );
    MatSetValuesLocal( K, 4, nodeList, 4, nodeList, tmpMem, ADD_VALUES );
  
    matrix->assembleLumpedMassMatrix( el_i, tmpMem );
    VecSetValuesLocal( lumped_mm, 4, nodeList, tmpMem, ADD_VALUES );
  
  }
  MatAssemblyBegin( M ,MAT_FINAL_ASSEMBLY); MatAssemblyEnd( M,MAT_FINAL_ASSEMBLY);
  MatAssemblyBegin( K ,MAT_FINAL_ASSEMBLY); MatAssemblyEnd( K,MAT_FINAL_ASSEMBLY);
  VecAssemblyBegin(lumped_mm); VecAssemblyEnd(lumped_mm);

  // build dirichlet bc indices using ao
  PetscInt *bc_global_id, appId;
  PetscScalar *bcVals;
  PetscMalloc1(2*ijk_sizes[0], &bc_global_id);
  PetscMalloc1(2*ijk_sizes[0], &bcVals);
  for( n_i=0; n_i<ijk_sizes[0]; n_i++ ){
    bc_global_id[n_i] = n_i;
    bcVals[n_i] = 1;
  }
  for( n_i=0; n_i<ijk_sizes[0]; n_i++ ){
    appId = ijk_sizes[0]*(ijk_sizes[1]-1)+n_i;
    bc_global_id[ijk_sizes[0]+n_i] = appId;
    bcVals[ijk_sizes[0]+n_i] = 0.1;
  }
  
  AO ao;
  DMDAGetAO(da,&ao);
  AOApplicationToPetsc(ao, 2*ijk_sizes[0], bc_global_id);
  VecSetValues(phi, 2*ijk_sizes[0], bc_global_id, bcVals, INSERT_VALUES);
  VecAssemblyBegin(phi); VecAssemblyEnd(phi);
  
  // copy dirichlet bc info to fields, used in implicit solves
  field->SetBCInfo(2*ijk_sizes[0], bc_global_id, bcVals);
  
  PetscFree(bc_global_id);
  PetscFree(bcVals);

  PetscPrintf(PETSC_COMM_WORLD,"Running %d steps with time interval %f\n", num_timeSteps, dt);
  PetscPrintf(PETSC_COMM_WORLD,"Time discretisation - ");
  if (solver_type == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"CN solver\n");
    SolveCN( field, &K, &M, &phi, num_timeSteps, dt ); // Seems stable
  } else if (solver_type == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"B. Euler solver\n");
    SolveBackwardEuler( field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );
  } else if (solver_type == 2) {
    PetscPrintf(PETSC_COMM_WORLD,"F. Euler solver\n");
    SolveForwardEuler( field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"''-solver_type' is unknown, not solving");
  }
  // SolvePredictorCorrector( field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );
  // SolveEuler_FE_LumpedMM( field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );


  // Generate gnuplot with sequential writes
  Vec xy;
  DMGetCoordinates(da,&xy);
  FILE *oFile=NULL;
  int number = 1;
  if (rank != 0) {
    // blocking mpi receive
    MPI_Recv( &number, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, NULL );
    oFile = fopen("field.out", "a");
  } else {
    oFile = fopen("field.out", "w");
  }
  
  int i,nLocal;
  double *coords, *vals;
  VecGetLocalSize(phi,&nLocal);
  VecGetArray(xy,&coords);
  VecGetArray(phi, &vals);
  for( i=0; i<nLocal; i++) {
    fprintf(oFile, "%f %f %f\n", coords[2*i], coords[2*i+1], vals[i]);
  }
  VecRestoreArray(xy,&coords);
  VecRestoreArray(phi,&vals);
  
  fclose(oFile);
  
  if (rank != nProcs-1) MPI_Send(&number, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);

  // PetscViewer viewer;
  // PetscViewerHDF5Open(PETSC_COMM_WORLD,"field.h5",FILE_MODE_WRITE,&viewer);
  // PetscViewerSetFromOptions(viewer);
  // VecView(phi, viewer);
  // PetscViewerDestroy(&viewer);
  // 
  // PetscViewerHDF5Open(PETSC_COMM_WORLD,"coords.h5",FILE_MODE_WRITE,&viewer);
  // PetscViewerSetFromOptions(viewer);
  // VecView(xy, viewer);
  // PetscViewerDestroy(&viewer);

  delete matrix;
  delete field;
  delete mesh;

  VecDestroy(&lumped_mm); 
  VecDestroy(&phi);
  MatDestroy(&M); 
  MatDestroy(&K);
  DMDestroy(&da);


  PetscMallocGetCurrentUsage( &petscMem );
  printf("*****\nPetscMalloc, after petscFinal %f\n*******\n", petscMem );
  PetscFinalize();
  return 0;
}

void SolvePredictorCorrector( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {
/*
[M] T_n+0.5 = [ M + 0.5 dt K ] T_n
[M] T_n+1   = [M + 1.0 dt K] T_n+0.5
*/ 

  Vec f_k, f_rhs;
  Mat A, B, C;
  KSP ksp;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );


  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &C ); MatZeroEntries( C );

  // Predictor operator
  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, 0.5*dt, *K, SAME_NONZERO_PATTERN );

  // Corrector operator
  MatAXPY( C, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( C, dt, *K, SAME_NONZERO_PATTERN );

  // get bc info
  int numBC, *nodeBCList;
  double *nodeBCValue;

  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  MatZeroRows( B, numBC, nodeBCList, 1.0, *phi, f_k);
  
  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,B,B);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi
    VecAXPY( f_rhs, 1, f_k );

    KSPSolve( ksp, f_rhs, *phi ); // T_n+0.5
    
    // Corrector
    VecScale(f_rhs, 0);
    MatMult(C, *phi, f_rhs);
    VecAXPY( f_rhs, 1, f_k );

    KSPSolve( ksp, f_rhs, *phi ); // T_n+0.5
    

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A ); MatDestroy(&B); MatDestroy(&C);
  KSPDestroy(&ksp);
}


void SolveForwardEuler( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {
/*
[M] T_n+1 = [ M + dt K ] T_n
*/ 

  Vec f_k, f_rhs;
  Mat A, B;
  KSP ksp;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );


  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, dt, *K, SAME_NONZERO_PATTERN );

  // get bc info
  int numBC, *nodeBCList;
  double *nodeBCValues;

  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValues );

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  
  MatZeroRows( B, numBC, nodeBCList, 1.0, *phi, f_k);

  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,B,B);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi

    VecAXPY( f_rhs, 1, f_k );

    KSPSolve( ksp, f_rhs, *phi );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A ); MatDestroy(&B);
  KSPDestroy(&ksp);
}

void SolveBackwardEuler( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {
/*
[ M - dt K ] T_n+1 = [ M ] T_n
*/ 

  Vec f_k, f_mm, f_rhs;
  Mat A, B;
  KSP ksp;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );
  VecDuplicate( *phi, &f_mm );


  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, -1*dt, *K, SAME_NONZERO_PATTERN );

  // get bc info
  int numBC, *nodeBCList;
  double *nodeBCValue;

  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  
  MatZeroRows(A, numBC, nodeBCList, 1.0, *phi, f_k );

  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,A,A);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

    MatMult( B, *phi, f_rhs ); // f_rhs = [A]*phi
    
    VecAXPY( f_rhs, 1, f_k );

    KSPSolve( ksp, f_rhs, *phi );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A ); MatDestroy(&B);
  KSPDestroy(&ksp);
}

void SolveEuler_FE_LumpedMM( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {
/*
DISABLED

[M_v] T_n+1 = [M + dt K] T_n
*/
  Vec f_k, f_rhs;
  Mat A;
  int numBC, *nodeBCList;
  double *nodeBCValue;
  
  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );
  VecSetValues( *phi, numBC, nodeBCList, nodeBCValue, INSERT_VALUES );
  VecAssemblyBegin(*phi); VecAssemblyEnd(*phi);

  VecDuplicate( *phi, &f_k );
  VecDuplicate( *phi, &f_rhs );

  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, dt, *K, SAME_NONZERO_PATTERN );

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi
    // VecAXPY( f_rhs, -1, f_k );
    assert(0);
    // NEED TO SET THE BCS IN THE LUMPED_MM vector
    
    VecPointwiseDivide( *phi, f_rhs, *lumped_mm );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A );
}

void SolveCN( Field* field, Mat *K, Mat *M, Vec *phi, int num_timeSteps, double dt ) {
/*
 [M - 0.5 dt K] T_n+1 = [M + 0.5 dt K] T_n
 */
 
  Vec f_k, f_rhs;
  Mat A, B;
  KSP ksp;
  int     numBC, *nodeBCList;
  double *nodeBCValue;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );

  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( A,       1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A,  0.5*dt, *K, SAME_NONZERO_PATTERN );

  // get bc info
  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );
  
  MatAXPY( B,       1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( B, -0.5*dt, *K, SAME_NONZERO_PATTERN );
  
  MatZeroRows(B, numBC, nodeBCList, 1.0, *phi, f_k );
  
  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,B,B);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {
        
    MatMult( A, *phi, f_rhs );
    VecAXPY( f_rhs, 1, f_k );

    KSPSolve( ksp, f_rhs, *phi );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A ); MatDestroy(&B);
  KSPDestroy( &ksp );
}

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( phi, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
#endif
