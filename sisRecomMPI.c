#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define ALEATORIO ((double)rand() / (double)RAND_MAX)

void preenche_aleatorio_LR(double *L, double *R, int nU, int nI, int nF)
{
    srand(0);
    for (int i = 0; i < nU; i++)
    {
        for (int j = 0; j < nF; j++)
        {
            L[i * nF + j] = ALEATORIO / (double)nF;
        }
    }

    for (int i = 0; i < nF; i++)
    {
        for (int j = 0; j < nI; j++)
        {
            R[i * nI + j] = ALEATORIO / (double)nF;
        }
    }
}

void multiplicacao(double *B, double *L, double *R, int nUL, int nIR, int nIL)
{
    for (int i = 0; i < nUL; i++)
    {
        for (int j = 0; j < nIR; j++)
        {
            B[i * nIR + j] = 0;
            for (int k = 0; k < nIL; k++)
            {
                B[i * nIR + j] += L[i * nIL + k] * R[k * nIR + j];
            }
        }
    }
}

void calcularL(double *L, double *A, double *B, double *R, double alfa, int nU, int nI, int nF, int rank, int size)
{
    double soma = 0.0;
    int chunk_size = (nU + size - 1) / size; // calcula tamanho do pedaço considerando resto
    int start = rank * chunk_size;
    int end = (start + chunk_size > nU) ? nU : start + chunk_size; // ajusta o fim se ultrapassar nU

    for (int i = start; i < end; i++)
    {
        for (int k = 0; k < nF; k++)
        {
            soma = 0.0;
            for (int j = 0; j < nI; j++)
            {
                if (A[i * nI + j] == 0)
                    continue;
                soma += (2 * (A[i * nI + j] - B[i * nI + j]) * (-R[k * nI + j]));
            }
            L[i * nF + k] = L[i * nF + k] - (alfa * soma);
        }
    }
}

void calcularR(double *R, double *A, double *B, double *L, double alfa, int nU, int nI, int nF)
{
    double soma = 0.0;
    for (int k = 0; k < nF; k++)
    {
        for (int j = 0; j < nI; j++)
        {
            soma = 0.0;
            for (int i = 0; i < nU; i++)
            {
                if (A[i * nI + j] == 0)
                    continue;
                soma += 2 * (A[i * nI + j] - B[i * nI + j]) * (-L[i * nF + k]);
            }
            R[k * nI + j] = R[k * nI + j] - (alfa * soma);
        }
    }
}

void inicializar(double *A, double *B, double *L, double *R, int nU, int nI, int nF)
{
    for (int i = 0; i < nU * nI; i++)
    {
        A[i] = 0.0;
    }

    for (int i = 0; i < nU * nF; i++)
    {
        L[i] = 0.0;
    }

    for (int i = 0; i < nF * nI; i++)
    {
        R[i] = 0.0;
    }
}
int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start = MPI_Wtime();
    int numero_iteracoes, nU, nI, nF, dZ, linha, coluna;
    double alfa, valor;

    FILE *ficheiro;
    if (rank == 0)
    {
        ficheiro = fopen(argv[1], "r");
        if (ficheiro == NULL)
        {
            printf("Erro ao abrir o arquivo.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(ficheiro, "%d", &numero_iteracoes);
        fscanf(ficheiro, "%lf", &alfa);
        fscanf(ficheiro, "%d", &nF);
        fscanf(ficheiro, "%d %d %d", &nU, &nI, &dZ);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&numero_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alfa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nF, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nU, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nI, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *A = (double *)malloc(nU * nI * sizeof(double));
    double *L = (double *)malloc(nU * nF * sizeof(double));
    double *L_auxiliar = (double *)malloc(nU * nF * sizeof(double));
    double *R = (double *)malloc(nF * nI * sizeof(double));
    double *B = (double *)malloc(nU * nI * sizeof(double));

    if (rank == 0)
    {
        inicializar(A, B, L, R, nU, nI, nF);
        for (int i = 0; i < dZ; i++)
        {
            fscanf(ficheiro, "%d %d %lf", &linha, &coluna, &valor);
            A[linha * nI + coluna] = valor;
        }
        preenche_aleatorio_LR(L, R, nU, nI, nF);
        multiplicacao(B, L, R, nU, nI, nF);
        fclose(ficheiro);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Bcast(&A[0], nU * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&L[0], nU * nF, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&R[0], nF * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&B[0], nU * nI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int iteracao = 1; iteracao <= numero_iteracoes; iteracao++)
    {
        for (int i = 0; i < nU * nF; i++)
        {
            L_auxiliar[i] = L[i];
        }

        calcularL(L, A, B, R, alfa, nU, nI, nF, rank, size);
        calcularR(R, A, B, L_auxiliar, alfa, nU, nI, nF);

        MPI_Allgather(MPI_IN_PLACE, (nU + size - 1) / size * nF, MPI_DOUBLE, L, (nU + size - 1) / size * nF, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgather(MPI_IN_PLACE, nF * nI, MPI_DOUBLE, R, nF * nI, MPI_DOUBLE, MPI_COMM_WORLD);

        multiplicacao(B, L, R, nU, nI, nF);
    }

    if (rank == 0)
    {
        FILE *saida = fopen("saida.txt", "w");
        if (saida == NULL)
        {
            printf("Erro ao abrir o arquivo de saída.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double maior = 0.0;
        int indice = 0;
        for (int i = 0; i < nU; i++)
        {
            maior = 0.0;
            for (int j = 0; j < nI; j++)
            {
                if (A[i * nI + j] == 0)
                {
                    indice = (B[i * nI + j] > maior) ? j : indice;
                    maior = (B[i * nI + j] > maior) ? B[i * nI + j] : maior;
                }
            }
            fprintf(saida, "%d\n", indice);
        }

        fclose(saida);
    }

    // Liberar memória alocada
    free(A);
    free(L);
    free(L_auxiliar);
    free(R);
    free(B);

    double end = MPI_Wtime();
    if (rank == 0)
    {
        printf("tempo %lf\n", end - start);
    }

    MPI_Finalize();
    return 0;
}