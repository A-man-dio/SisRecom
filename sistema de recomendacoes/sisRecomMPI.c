#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ALEATORIO ((double)rand() / (double)RAND_MAX)

void preenche_aleatorio_LR(double **L, double **R, int nU, int nI, int nF)
{
    srand(0);
    int i, j;
    for (i = 0; i < nU; i++)
    {
        for (j = 0; j < nF; j++)
        {
            L[i][j] = ALEATORIO / (double)nF;
        }
    }

    for (i = 0; i < nF; i++)
    {
        for (j = 0; j < nI; j++)
        {
            R[i][j] = ALEATORIO / (double)nF;
        }
    }
}

void imprimir_matriz(double **matriz, int numero_linhas, int numero_colunas)
{
    for (int i = 0; i < numero_linhas; i++)
    {
        for (int j = 0; j < numero_colunas; j++)
        {
            printf("%lf ", matriz[i][j]);
        }
        printf("\n");
    }
}

void calcular_B(double **B, double **L, double **R, int numero_linhasL, int numero_colunasR, int numero_colunasL)
{
    for (int i = 0; i < numero_linhasL; i++)
    {
        for (int j = 0; j < numero_colunasR; j++)
        {
            B[i][j] = 0;
            for (int k = 0; k < numero_colunasL; k++)
            {
                B[i][j] += L[i][k] * R[k][j];
            }
        }
    }
}

void calcular_L_posterior(double **L, double **A, double **B, double **R, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas, int rank, int size)
{
    int linhas_por_processo = numero_linhas / size;
    int inicio = rank * linhas_por_processo;
    int fim = (rank == size - 1) ? numero_linhas : inicio + linhas_por_processo;

    double soma = 0;

    for (int i = inicio; i < fim; i++)
    {
        for (int k = 0; k < numero_caracteristicas; k++)
        {   
            for (int j = 0; j < numero_colunas; j++)
            {
                soma = (A[i][j] != 0) ? soma + (2 * (A[i][j] - B[i][j]) * (-R[k][j])) : soma;
            }
            L[i][k] = L[i][k] - (alfa * soma);
            soma = 0;
        }
    }
}

void calcular_R_posterior(double **R, double **A, double **B, double **L, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas, int rank, int size)
{
    int colunas_por_processo = numero_colunas / size;
    int inicio = rank * colunas_por_processo;
    int fim = (rank == size - 1) ? numero_colunas : inicio + colunas_por_processo;

    double soma = 0;

    for (int k = 0; k < numero_caracteristicas; k++)
    {
        for (int j = inicio; j < fim; j++)
        {
            for (int i = 0; i < numero_linhas; i++)
            {
                soma = (A[i][j] != 0) ? soma + (2 * (A[i][j] - B[i][j]) * (-L[i][k])) : soma;
            }
            R[k][j] = R[k][j] - (alfa * soma);
            soma = 0;
        }
    }
}

void inicializar_matriz(double **matriz, int numero_linhas, int numero_colunas)
{
    for (int i = 0; i < numero_linhas; i++)
    {
        for (int j = 0; j < numero_colunas; j++)
        {
            matriz[i][j] = 0.0;
        }
    }
}


int main(int argc, char *argv[])
{
    int rank, size;

    // Inicializar o MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    FILE *input_file;
    int numero_iteracoes, numero_linhas, numero_colunas, numero_caracteristicas, numero_elementos_diferentes_de_zero;
    int linha = 0, coluna = 0;
    double alfa, valor = 0;

    if (rank == 0)
    {
        input_file = fopen("input/dado0.in", "r");

        if (input_file == NULL)
        {
            printf("Erro ao abrir o arquivo.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        fscanf(input_file, "%d", &numero_iteracoes);
        fscanf(input_file, "%lf", &alfa);
        fscanf(input_file, "%d", &numero_caracteristicas);
        fscanf(input_file, "%d %d %d", &numero_linhas, &numero_colunas, &numero_elementos_diferentes_de_zero);

        // Broadcast de parâmetros
        MPI_Bcast(&numero_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&alfa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_caracteristicas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_linhas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_colunas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_elementos_diferentes_de_zero, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Criação e inicialização da matriz de entrada A
        double matriz = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            matriz[i] = (double *)malloc(numero_colunas * sizeof(double));
        }

        inicializar_matriz(matriz, numero_linhas, numero_colunas);

        // Preenchimento da matriz de entrada com os dados do ficheiro dado0.in
        for (int i = 0; i < numero_elementos_diferentes_de_zero; i++)
        {
            fscanf(input_file, "%d %d %lf", &linha, &coluna, &valor);
            matriz[linha][coluna] = valor;
        }

        fclose(input_file);

        // Broadcast da matriz A
        for (int i = 0; i < numero_linhas; i++)
        {
            MPI_Bcast(matriz[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Inicializar matriz L
        double L = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            L[i] = (double *)malloc(numero_caracteristicas * sizeof(double));
        }

        // Inicializar matriz R
        double R = (double )malloc(numero_caracteristicas * sizeof(double *));
        for (int i = 0; i < numero_caracteristicas; i++)
        {
            R[i] = (double *)malloc(numero_colunas * sizeof(double));
        }

        inicializar_matriz(R, numero_caracteristicas, numero_colunas);
        inicializar_matriz(L, numero_linhas, numero_caracteristicas);

        preenche_aleatorio_LR(L, R, numero_linhas, numero_colunas, numero_caracteristicas);

        // Broadcast das matrizes L e R inicializadas
        for (int i = 0; i < numero_linhas; i++)
        {
            MPI_Bcast(L[i], numero_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < numero_caracteristicas; i++)
        {
            MPI_Bcast(R[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Criação da matriz B = L * R
        double B = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            B[i] = (double *)malloc(numero_colunas * sizeof(double));
        }
        inicializar_matriz(B, numero_linhas, numero_colunas);

        calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);

        // Início das iterações
        for (int iter = 1; iter <= numero_iteracoes; iter++)
        {
            calcular_L_posterior(L, matriz, B, R, alfa, numero_linhas, numero_colunas, numero_caracteristicas, rank, size);
            calcular_R_posterior(R, matriz, B, L, alfa, numero_linhas, numero_colunas, numero_caracteristicas, rank, size);
// Gather para L e R após cada iteração
            for (int i = 0; i < numero_linhas; i++)
            {
                MPI_Bcast(L[i], numero_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            for (int i = 0; i < numero_caracteristicas; i++)
            {
                MPI_Bcast(R[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);
        }

        // Impressão dos resultados finais
        printf("\nB:\n");
        imprimir_matriz(B, numero_linhas, numero_colunas);
        printf("\nR:\n");
        imprimir_matriz(R, numero_caracteristicas, numero_colunas);
        printf("\nL:\n");
        imprimir_matriz(L, numero_linhas, numero_caracteristicas);

        // Liberação da memória
        for (int i = 0; i < numero_linhas; i++)
        {
            free(matriz[i]);
            free(L[i]);
            free(B[i]);
        }
        for (int i = 0; i < numero_caracteristicas; i++)
        {
            free(R[i]);
        }
        free(matriz);
        free(L);
        free(R);
        free(B);
    }
    else
    {
        // Receber os parâmetros via broadcast
        MPI_Bcast(&numero_iteracoes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&alfa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_caracteristicas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_linhas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_colunas, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&numero_elementos_diferentes_de_zero, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Receber a matriz A via broadcast
        double matriz = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            matriz[i] = (double *)malloc(numero_colunas * sizeof(double));
            MPI_Bcast(matriz[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Inicializar matriz L
        double L = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            L[i] = (double *)malloc(numero_caracteristicas * sizeof(double));
            MPI_Bcast(L[i], numero_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Inicializar matriz R
        double R = (double )malloc(numero_caracteristicas * sizeof(double *));
        for (int i = 0; i < numero_caracteristicas; i++)
        {
            R[i] = (double *)malloc(numero_colunas * sizeof(double));
            MPI_Bcast(R[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Criação da matriz B = L * R
        double B = (double )malloc(numero_linhas * sizeof(double *));
        for (int i = 0; i < numero_linhas; i++)
        {
            B[i] = (double *)malloc(numero_colunas * sizeof(double));
        }
        inicializar_matriz(B, numero_linhas, numero_colunas);

        calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);

        // Início das iterações
        for (int iter = 1; iter <= numero_iteracoes; iter++)
        {
            calcular_L_posterior(L, matriz, B, R, alfa, numero_linhas, numero_colunas, numero_caracteristicas, rank, size);
            calcular_R_posterior(R, matriz, B, L, alfa, numero_linhas, numero_colunas, numero_caracteristicas, rank, size);

            // Gather para L e R após cada iteração
            for (int i = 0; i < numero_linhas; i++)
            {
                MPI_Bcast(L[i], numero_caracteristicas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            for (int i = 0; i < numero_caracteristicas; i++)
            {
                MPI_Bcast(R[i], numero_colunas, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            }

            calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);
        }
// Liberação da memória
        for (int i = 0; i < numero_linhas; i++)
        {
            free(matriz[i]);
            free(L[i]);
            free(B[i]);
        }
        for (int i = 0; i < numero_caracteristicas; i++)
        {
            free(R[i]);
        }
        free(matriz);
        free(L);
        free(R);
        free(B);
    }

    // Finalizar o MPI
    MPI_Finalize();

    return 0;
}
