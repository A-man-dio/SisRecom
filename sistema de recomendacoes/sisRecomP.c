#include <stdio.h>
#include <stdlib.h>
//#include <omp.h>
#define ALEATORIO ((double)rand() / (double)RAND_MAX)

void preenche_aleatorio_LR(double **L, double **R, int nU, int nI, int nF)
{
    srand(0);
    #pragma omp paralell{
    
    #pragma omp for
    for (int i = 0; i < nU; i++)
    {
        for (int j = 0; j < nF; j++)
        {
            L[i][j] = ALEATORIO / (double)nF;
        }
    }

    #pragma omp for
    for (int i = 0; i < nF; i++)
    {
        for (int j = 0; j < nI; j++)
        {
            R[i][j] = ALEATORIO / (double)nF;
        }
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
    #pragma omp paralell for
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

void calcular_L_posterior(double **L, double **A, double **B, double **R, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas)
{
    double soma = 0;
    
    #pragma omp paralell for private(soma)
    for (int i = 0; i < numero_linhas; i++)
    {
        for (int k = 0; k < numero_caracteristicas; k++)
        {   
            for (int j = 0; j < numero_colunas; j++)
            {
                // soma += (2 * (A[i][j] - B[i][j]) * (-R[k][j]));
                soma = (A[i][j] != 0) ? soma + (2 * (A[i][j] - B[i][j]) * (-R[k][j])) : soma;
            }
            L[i][k] = L[i][k] - (alfa * soma);
            soma = 0;
        }
    }
}

void calcular_R_posterior(double **R, double **A, double **B, double **L, double alfa, int numero_linhas, int numero_colunas, int numero_caracteristicas)
{
    double soma = 0;
    #pragma omp paralell for private (soma)
    for (int k = 0; k < numero_caracteristicas; k++)
    {
        for (int j = 0; j < numero_colunas; j++)
        {
            for (int i = 0; i < numero_linhas; i++)
            {
                // soma += 2 * (A[i][j] - B[i][j]) * (-L[i][k]);
                soma = (A[i][j] != 0) ? soma + (2 * (A[i][j] - B[i][j]) * (-L[i][k])) : soma;
            }
            R[k][j] = R[k][j] - (alfa * soma);
            soma = 0;
        }
    }
}

void inicializar_matriz(double **matriz, int numero_linhas, int numero_colunas)
{
    #pragma omp paralell for
    for (int i = 0; i < numero_linhas; i++)
    {
        for (int j = 0; j < numero_colunas; j++)
        {
            matriz[i][j] = 0.0;
        }
    }
}

int main()
{
    FILE *input_file;
    int numero_iteracoes, numero_linhas=0, numero_colunas, numero_caracteristicas, numero_elementos_diferentes_de_zero, linha = 0, coluna = 0;
    double alfa, valor = 0;
    int cont=0;
    double start_time = omp_get_wtime();
    input_file = fopen("dado30-40-10-2-10.in", "r");

    if (input_file == NULL)
    {
        printf("Erro ao abrir o arquivo.\n");
        exit(1);
    }

    omp_set_num_threads(4);

    #pragma omp parallel{
        #pragma omp sections{
            #pragma omp section
            fscanf(input_file, "%d", &numero_iteracoes);
            #pragma omp section
            fscanf(input_file, "%lf", &alfa);
            #pragma omp section
            fscanf(input_file, "%d", &numero_caracteristicas);
            #pragma omp section
            fscanf(input_file, "%d %d %d", &numero_linhas, &numero_colunas, &numero_elementos_diferentes_de_zero);
        }
        if((numero_linhas==0) &&(cont==0)){
        cont =1;
        #pragma omp barrier
        }
    }
        // criação e inicialização das matrizes 
        double **matriz = (double **)malloc(numero_linhas * sizeof(double *));
        double **L = (double **)malloc(numero_linhas * sizeof(double *));
        double **R = (double **)malloc(numero_caracteristicas * sizeof(double *));
        double **B = (double **)malloc(numero_linhas * sizeof(double *));

        #pragma omp parallel{
            #pragma omp for
            for (int i = 0; i < numero_linhas; i++)
                matriz[i] = (double *)malloc(numero_colunas * sizeof(double));

            #pragma omp for
            for (int i = 0; i < numero_linhas; i++)
                L[i] = (double *)malloc(numero_caracteristicas * sizeof(double));
       
            #pragma omp for
            for (int i = 0; i < numero_caracteristicas; i++)
                R[i] = (double *)malloc(numero_colunas * sizeof(double));
            
            #pragma omp for
            for (int i = 0; i < numero_linhas; i++)
            B[i] = (double *)malloc(numero_colunas * sizeof(double));
        }

        inicializar_matriz(matriz, numero_linhas, numero_colunas);
        inicializar_matriz(R, numero_caracteristicas, numero_colunas);
        inicializar_matriz(L, numero_linhas, numero_caracteristicas);
        inicializar_matriz(B, numero_linhas, numero_colunas);

    // preenchimento da matriz de entrada com os dados do ficheiro dado.in
    #pragma omp parallel for private(linha,coluna,valor)
    for (int i = 0; i < numero_elementos_diferentes_de_zero; i++)
    {
        fscanf(input_file, "%d %d %lf", &linha, &coluna, &valor);
        matriz[linha][coluna] = valor;
    }

    preenche_aleatorio_LR(L, R, numero_linhas, numero_colunas, numero_caracteristicas);

    calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);

    
    // começo da iteração

    for (int i = 1; i <= numero_iteracoes; i++)
    {
        calcular_L_posterior(L, matriz, B, R, alfa, numero_linhas, numero_colunas, numero_caracteristicas);
        calcular_R_posterior(R, matriz, B, L, alfa, numero_linhas, numero_colunas, numero_caracteristicas);
        calcular_B(B, L, R, numero_linhas, numero_colunas, numero_caracteristicas);
    }

    // Após isso teremos o B(5000)

    // agora iremos pegar para cada linha o índice do item de maior valor da matriz B , sem levar em consideração aqueles já avaliados na matriz de entrada (A)

    
    #pragma omp paralell for
    for (int i = 0; i < numero_linhas; i++)
    {
        double maior = 0.0;
    int indice = 0;
        maior = 0.0;
        for (int j = 0; j < numero_colunas; j++)
        {
            if (matriz[i][j] == 0)
            {
                indice = (B[i][j] > maior) ? j : indice;
                maior = (B[i][j] > maior) ? B[i][j] : maior;
            }
        }
        printf("%d\n", indice);
    }

    double end_time = omp_get_wtime();
    printf("\n\n");
   printf("\nB:\n");
    imprimir_matriz(B,numero_linhas,numero_colunas);
    printf("\nR:\n");
    imprimir_matriz(R,numero_caracteristicas,numero_colunas);
    printf("\nL:\n");
    imprimir_matriz(L,numero_linhas,numero_caracteristicas);

    printf("\n\nTempo final: %lf\n\n",(end_time-start_time));
    return (0);
}