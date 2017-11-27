include /usr/local/petsc-3.8.0/lib/petsc/conf/variables

CFLAGS = ${PETSC_CC_INCLUDES} ${CXX_FLAGS} ${CXXFLAGS} ${CCPPFLAGS}  ${PSOURCECXX}
CC = ${CXX}

femGame: FEMEngine.o Mesh.o Field.o Matrix.o
	${CC} -o femGame FEMEngine.o Matrix.o Field.o Mesh.o ${PETSC_LIB} ${CFLAGS}
FEMEngine.o: FEMEngine.cpp Matrix.o Field.o Mesh.o
	${CC} -c FEMEngine.cpp ${CFLAGS}
Matrix.o: Matrix.cpp Matrix.h Field.o Mesh.o
	${CC} -c Matrix.cpp ${CFLAGS}
Field.o: Field.cpp Field.h Mesh.o
	${CC} -c Field.cpp ${CFLAGS}
Mesh.o: Mesh.cpp Mesh.h
	${CC} -c Mesh.cpp ${CFLAGS}
clean:
	rm *.o femGame
