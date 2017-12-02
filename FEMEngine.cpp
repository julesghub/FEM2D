#include<iostream>

#include <petsc.h>
#include <petscdm.h>
#include "petscsys.h"   
#include "petscviewerhdf5.h"


#include "Mesh.h"
#include "Field.h"
#include "Matrix.h"

static char help[] = "My fist petsc try.\n";

void SergioCorrectForDirchletBC(
    Mat *A,
    Vec *rhs,
    Field *field,
    Vec *f_dbc );

void CorrectVectorForDirchletBC(
    Mat *A,
    Vec *rhs,
    Field* field,
    Vec *f_dbc );

void SolveEuler_Lumped(
    Field *field,
    Mat *K, Mat *M,
     Vec *lumped_mm, Vec *phi,
    int num_timeSteps, double dt );

void SolveEuler_FullMM(
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
  double dt, minSep;
  double tmpMem[16];
  int    elRes[] = {10,10};
  int    num_timeSteps;
  int    rank, nProcs;
  int    ijk_sizes[3], n_i;
  
  PetscBool flg;
  PetscLogDouble petscMem;
  
  // start up petsc
  PetscInitialize( &argc, &argv, (char*)0, help );
  MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
  MPI_Comm_size(PETSC_COMM_WORLD,&nProcs);
  
  // get cmd args 
  PetscOptionsGetInt(PETSC_NULL, PETSC_NULL, "-n", &num_timeSteps, &flg);
  if( !flg ) num_timeSteps = 1; 
  PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-max_x", &max[0], &flg);
  PetscOptionsGetScalar(PETSC_NULL, PETSC_NULL, "-max_y", &max[1], &flg);

  // create dm
  DMDACreate2d(PETSC_COMM_WORLD,
               DM_BOUNDARY_NONE,DM_BOUNDARY_NONE, /* boundary types in x/y */
               DMDA_STENCIL_BOX,                  /* neighbour shape */
               elRes[0]+1, elRes[1]+1,            /* grid size */
               PETSC_DECIDE,PETSC_DECIDE,         /* nprocs in each direction */
               1,1,                               /* dof and width */
               NULL, NULL, &da);                  /* nodes along x & y */
  DMSetFromOptions(da);
  DMSetUp(da); // MUST CALL!
  DMDAGetInfo(da,0,&ijk_sizes[0],&ijk_sizes[1],&ijk_sizes[2],0,0,0,0,0,0,0,0,0);
  int _lElNum, lElNum, _nodesPerEl, nodesPerEl;
  const int *_e_n_graph;
  DMDAGetElements( da, &_lElNum, &_nodesPerEl, &_e_n_graph );
  lElNum = _lElNum;
  nodesPerEl = _nodesPerEl;
  DMDARestoreElements(da, &_lElNum, &_nodesPerEl, &_e_n_graph);
  
  Mesh* mesh = new Mesh( &da, ijk_sizes[0], ijk_sizes[1], max[0], max[1] );
  Field* field   = new Field(mesh, 1);
  Matrix* matrix = new Matrix(mesh, field, "matrix1");

  PetscSynchronizedPrintf(PETSC_COMM_WORLD, "I'm %d: I have %d local elements\n", rank, lElNum);
  PetscSynchronizedFlush(PETSC_COMM_WORLD, NULL);

  // calculate timesetp
  minSep = min_double( (max[0]/ijk_sizes[0]),  (max[1]/ijk_sizes[1]) );
  dt = 0.5*(minSep*minSep);

  PetscMallocGetCurrentUsage( &petscMem );
  printf("*****\nPetscMalloc, before run %f\n*******\n", petscMem );
  
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
  
  // copy dirichlet bc info to fields, used in solvers
  field->SetBCInfo(2*ijk_sizes[0], bc_global_id, bcVals);
  
  PetscFree(bc_global_id);
  PetscFree(bcVals);
//
// /* Need a very small timestep for Euler_FullMM. ~0.05
//   SolveEuler_FullMM( &field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );
// */
//
// /* CourantFactor ~ 0.6
//   SolveEuler_Lumped( &field, &K, &M, &lumped_mm, &phi, num_timeSteps, dt );
// */
//   //
//   // Seems stable
  SolveCN( field, &K, &M, &phi, num_timeSteps, dt );
//

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
  // DMDAGetCoordinateArray(da, coords );
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

void SergioCorrectForDirchletBC(
    Mat *A,
    Vec *rhs,
    Field *field,
     Vec *f_dbc  ) {

  int ii, jj;
  int row;
  PetscInt nodeCount;
  int *nodeBCList;
  double *nodeBCValue;
  int numBC;

  // get BC info
  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );

  /** initialise f_dbc */
  VecScale( *f_dbc, 0 );

  VecGetSize( *rhs, &nodeCount );

  /** knock out BC dofs from Matrix and rhs */
  for( ii = 0 ; ii < numBC ; ii++ ) {
    row = nodeBCList[ii];
    /** zero everything in row */
    for( jj = 0 ; jj < nodeCount ; jj++ )
      MatSetValue( *A, row, jj, 0, INSERT_VALUES );

    /** set diagonal to one */
    MatSetValue( *A, row, row, 1, INSERT_VALUES );

    /** set rhs to BC */
    VecSetValue( *rhs, row, nodeBCValue[ii], INSERT_VALUES );
  }

  MatAssemblyBegin( *A ,MAT_FINAL_ASSEMBLY); MatAssemblyEnd( *A ,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin( *rhs ); VecAssemblyEnd( *rhs );

}

void CorrectVectorForDirchletBC(
    Mat *A,
    Vec *rhs,
    Field *field,
    Vec *f_dbc ) {

  int ii, jj;
  int row;
  double fac;
  PetscInt nodeCount;
  Mat A_;
  int *nodeBCList;
  double *nodeBCValue;
  int numBC;

  // get bc info
  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );
  /** initialise f_dbc */
  VecScale( *f_dbc, 0 );

  VecGetSize( *rhs, &nodeCount );

  MatZeroRows( *A, numBC, nodeBCList, 1.0, 0, 0);
  /** knock out BC dofs from Matrix and rhs */
  for( ii = 0 ; ii < numBC ; ii++ ) {
    int row = nodeBCList[ii];
    /** zero everything in row */
    // for( int jj = 0 ; jj < nodeCount ; jj++ )
    //   MatSetValue( *A, row, jj, 0, INSERT_VALUES );
    //
    // /** set diagonal to one */
    // MatSetValue( *A, row, row, 1, INSERT_VALUES );

    /** set rhs to BC */
    VecSetValue( *rhs, row, 0, INSERT_VALUES );
  }

  MatAssemblyBegin( *A ,MAT_FINAL_ASSEMBLY); MatAssemblyEnd( *A ,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin( *rhs ); VecAssemblyEnd( *rhs );

  /* create a clone A_ = A for reading values */
  MatDuplicate( *A, MAT_COPY_VALUES, &A_ );

  /** now swep through all other mat and vec corrections
   * needed to correct the rhs and 0 out mat columns */
  for( ii = 0 ; ii < numBC ; ii++ ) {
    row = nodeBCList[ii];
    for( jj = 0 ; jj < nodeCount ; jj++ ) {
      /* for each column containing BC, take mat entry and move it to rhs, then zero column */

      if( jj == row ) continue; /* already done */

      MatGetValues( A_, 1, &jj, 1, &row, &fac );
      if( fac == 0 ) continue;
      VecSetValue( *f_dbc , jj, -1*fac*nodeBCValue[ii], ADD_VALUES ); /* add correction */
      MatSetValue( *A, jj, row, 0, INSERT_VALUES );
    }
  }

  MatAssemblyBegin( *A ,MAT_FINAL_ASSEMBLY); MatAssemblyEnd( *A ,MAT_FINAL_ASSEMBLY);

  VecAssemblyBegin( *f_dbc ); VecAssemblyEnd( *f_dbc );

  MatDestroy( &A_ );
}

void SolveEuler_FullMM( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {

  Vec f_k, f_mm, f_rhs;
  Mat A, B;
  KSP ksp;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );
  VecDuplicate( *phi, &f_mm );

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( *K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nM\n"); MatView( *M, PETSC_VIEWER_STDOUT_SELF );
#endif

  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, -1*dt, *K, SAME_NONZERO_PATTERN );

  CorrectVectorForDirchletBC( &A, phi, field, &f_k );
  //MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY);

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  CorrectVectorForDirchletBC( &B, phi, field, &f_mm );
  //MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY);

  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,B,B);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( phi, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
#endif
    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi
    VecAXPY( f_rhs, -1, f_k );
    VecAXPY( f_rhs, 1, f_mm );

#ifdef DEBUG_PRINT
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nf_rhs\n"); VecView( f_rhs, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( *phi, PETSC_VIEWER_STDOUT_SELF );
#endif
    KSPSolve( ksp, f_rhs, *phi );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A ); MatDestroy(&B);
  KSPDestroy(&ksp);
}

void SolveEuler_Lumped( Field* field, Mat *K, Mat *M, Vec *lumped_mm, Vec *phi, int num_timeSteps, double dt ) {

  Vec f_k, f_rhs;
  Mat A;

  VecDuplicate( *phi, &f_k );
  VecDuplicate( *phi, &f_rhs );

#ifdef DEBUG_PRINT
    printf("\n\nM\n"); MatView( M, PETSC_VIEWER_STDOUT_SELF );
#endif

  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, -1*dt, *K, SAME_NONZERO_PATTERN );

  //MatAXPY( A, -1*dt, K, SAME_NONZERO_PATTERN );
  MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY);

  CorrectVectorForDirchletBC( &A, phi, field, &f_k );

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( phi, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nf_rhs\n"); VecView( f_rhs, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( *phi, PETSC_VIEWER_STDOUT_SELF );
#endif
    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi
    VecAXPY( f_rhs, -1, f_k );

    VecPointwiseDivide( *phi, f_rhs, *lumped_mm );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs);
  MatDestroy( &A );
}

void SolveCN( Field* field, Mat *K, Mat *M, Vec *phi, int num_timeSteps, double dt ) {

  Vec f_k, f_mm, f_rhs;
  Mat A, B;
  KSP ksp;

  VecDuplicate( *phi, &f_rhs );
  VecDuplicate( *phi, &f_k );
  VecDuplicate( *phi, &f_mm );

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( *K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nM\n"); MatView( *M, PETSC_VIEWER_STDOUT_SELF );
#endif

  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( A, -0.5*dt, *K, SAME_NONZERO_PATTERN );

  /*
  Set phi bc values correctly
  Use MatZeroRows
  */
  // get bc info
  int numBC, *nodeBCList;
  double *nodeBCValue;

  field->GetBCInfo( &numBC, &nodeBCList, &nodeBCValue );
  VecSetValues( *phi, numBC, nodeBCList, nodeBCValue, INSERT_VALUES );
  VecAssemblyBegin(*phi); VecAssemblyEnd(*phi);
  
  MatZeroRows( A, numBC, nodeBCList, 1.0, *phi, f_k);
  // CorrectVectorForDirchletBC( &A, phi, field, &f_k );
  //MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY);

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( B, +0.5*dt, *K, SAME_NONZERO_PATTERN );
  MatZeroRows( B, numBC, nodeBCList, 1.0, *phi, f_mm);
  // CorrectVectorForDirchletBC( &B, phi, field, &f_mm );
  //MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY);

  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,B,B);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

#ifdef DEBUG_PRINT
    printf("\n\nK\n"); MatView( K, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( phi, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
#endif
    MatMult( A, *phi, f_rhs ); // f_rhs = [A]*phi
    VecAXPY( f_rhs, -1, f_k );
    VecAXPY( f_rhs, 1, f_mm );

#ifdef DEBUG_PRINT
    printf("\n\nA\n"); MatView( A, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nf_rhs\n"); VecView( f_rhs, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nphi\n"); VecView( *phi, PETSC_VIEWER_STDOUT_SELF );
#endif
    KSPSolve( ksp, f_rhs, *phi );

  }
  VecDestroy( &f_k ); VecDestroy(&f_rhs); VecDestroy(&f_mm);
  MatDestroy( &A ); MatDestroy(&B);
  KSPDestroy( &ksp );
}
#if 0
  /** This uses CN to discretise the time stepping */
  KSP ksp;
  Mat A, B; /* A is op for n+1, B is op for n */
  Vec f_A, f_B, f_mm, f_k;

  VecDuplicate( *phi, &f_A ); VecDuplicate( *phi, &f_B );
  VecDuplicate( *phi, &f_k ); VecDuplicate( *phi, &f_mm );


  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &A ); MatZeroEntries( A );
  MatDuplicate( *K, MAT_DO_NOT_COPY_VALUES, &B ); MatZeroEntries( B );

  MatAXPY( B, 1, *M, SAME_NONZERO_PATTERN );
  MatAXPY( B, -1*dt, *K, SAME_NONZERO_PATTERN );
  MatAssemblyBegin( B, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( B, MAT_FINAL_ASSEMBLY);

  MatAXPY( A, 1, *M, SAME_NONZERO_PATTERN );
  //MatAXPY( A, 0.5*dt, *K, SAME_NONZERO_PATTERN );
  MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY); MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY);

  CorrectVectorForDirchletBC( &B, phi, field, &f_k );
  CorrectVectorForDirchletBC( &A, phi, field, &f_mm );

  KSPCreate( PETSC_COMM_WORLD, &ksp );
  KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);
  KSPSetFromOptions(ksp);

  for( int time_i = 0 ; time_i < num_timeSteps ; time_i++  ) {

#ifdef DEBUG_PRINT
    printf("\n\nf_rhs\n"); VecView( f_B, PETSC_VIEWER_STDOUT_SELF );
#endif
    MatMult( B, *phi, f_B ); // f_B = [B]*phi
    VecAXPY( f_B, -1, f_k );
    //VecAXPY( f_B, 0, f_mm );

    printf("\n\nB\n"); MatView( B, PETSC_VIEWER_STDOUT_SELF );
    printf("\n\nf_B\n"); VecView( f_B, PETSC_VIEWER_STDOUT_SELF );
    KSPSolve( ksp, f_B, *phi );
    printf("\n\nphi\n"); VecView( *phi, PETSC_VIEWER_STDOUT_SELF );

    /*
    VecScale( prev_dphi, 0 );
    for( int lump_it = 3 ; lump_it < 4 ; lump_it++ ) {

      // err_rhs = f_rhs - [M] prev_dphi
      MatMult( M, prev_dphi, err_rhs);
      VecScale( err_rhs, -1 );
      VecAXPY( err_rhs, 1, f_rhs );

      printf("\n\nerr_rhs\n"); VecView( err_rhs, PETSC_VIEWER_STDOUT_SELF );
      printf("\n\nlumped_mm\n"); VecView( lumped_mm, PETSC_VIEWER_STDOUT_SELF );
      // err_dphi = err_rhs / lumped_mm
      VecPointwiseDivide( err_dphi, err_rhs, lumped_mm );

      // dphi = err_dphi + prev_dphi
      VecAXPY( err_dphi, 1, prev_dphi );
      VecCopy( err_dphi, dphi );
      printf("\n\ndphi\n"); VecView( dphi, PETSC_VIEWER_STDOUT_SELF );

      VecCopy( dphi, prev_dphi );
    }
    */
    //VecAXPY( phi, 1, dphi );

  }
  KSPDestroy( ksp );
  MatDestroy( A ); MatDestroy( B );
  VecDestroy( f_A ); VecDestroy( f_B ); VecDestroy( f_mm ); VecDestroy( f_k);
}
#endif
