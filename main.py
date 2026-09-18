import sympy as sp

"""O programa consiste em receber um intervalo [a, b] onde irá aplicar o método de Newton e da Dicotomia
para encontrar zeros de funções, porém, antes de aplicar Newton ou Dicotomia ele precisa fazer algumas verificações
para decidir qual método aplicar, de acordo com o programa.
O fluxograma funciona da seguinte forma:
-> Recebe um intervalo [a, b] SEMPRE satisfazendo f(a) * f(b) < 0
-> Inicialmente decidimos começar aplicando o método da Dicotomia pois para aplicarmos Newton é necessário
que a condição (3) seja satisfeita e como no 1° passo xn+1 - xn não existe pois ainda não aplicamos xn+1, optams por isso
-> Então começamos aplicando dicotomia, depois antes de aplicar Newton checamos se a condição (1) e (2) são satisfeitas, se sim, aplicamos Newton e vamos pro prox

"""
#=== 1. FUNCOES AUXILIARES (calcular derivada, checagem de condicoes, newton etc)===
#funcao responsavel para calcular a derivada da funcao colocada pelo usuario, foi utilizada a biblioteca
#sympy para o calculo da derivada, e obrigatorio que a funcao a ser colocada esteja em funcao de 1 variavel
def calcula_derivada(func):
    try:
        x = sp.symbols('x')
        expressao = sp.sympify(func)
        expressao_derivada = sp.diff(expressao, x)

        f = sp.lambdify(x, expressao)
        f_linha = sp.lambdify(x, expressao_derivada)
        return f, f_linha
    except ValueError:
        print("Funcao esta incorreta")

#funcao responsavel por checar se f(a) * f(b) < 0 para todo passo, em caso negativo nao ha raiz 
def sinais_opostos(a, b, f):
    if f(a) * f(b) < 0: return True
    return False

#funcao que checa a condicao (2) do ep
def checa_condicao2(a, b, f, f_linha, xn):
    if ((xn - a) * f_linha(xn) - f(xn)) * ((xn - b) * f_linha(xn) - f(xn)) < 0:
        return True
    return False

#funcao que checa a condicao (3) do ep
def checa_condicao3(f, f_linha, xn, deltaxn):
    if 2 * abs(f(xn)) < abs(f_linha(xn) * deltaxn):
        return True
    return False

#funcao responsavle para aproximar a raiz utilizando o metodo de newton
def newton(xn, f, f_linha):
    return xn - f(xn)/f_linha(xn)

#funcao responsavle para aproximar a raiz utilizando o metodo da dicotomia
def dicotomia(a, b):
    xn = (a + b) / 2
    return xn 

#funcao que verifica se a condicao |Δxn| < ATOL + RTOL*|xn| é verdadeira, pois se sim, o codigo para
def sinais_quebra_1(deltaxn, xn, ATOL, RTOL):
    return abs(deltaxn) < ATOL + RTOL * abs(xn)

#funcao responsavel por atualizar o intervalo a cada interacao
def atualiza_intervalo(a, b, f, xn):
    if f(a) * f(xn) < 0:
        # raiz está entre alpha e x_novo
        return [a, xn]
    else:
        # raiz está entre x_novo e beta
        return [xn, b]


# === 2. O METODO EM SI, COMO FUNCAO REUTILIZAVEL =====
def newton_modificado(f, f_linha, a, b, ATOL, RTOL, MAXIT, verbose = True):
    """
    Método recebe tudo já pronto e devolve a raiz aproximada + historico de iteracoes
    """
    #definimos o intervalo inicial e logo depois atualizamos antes da primeira interacao, de modo a deixar
    #sempre mais proximo da raiz
    intervalo = [a, b]
    if abs(f(a)) < abs(f(b)):
        xn = a
        intervalo = [xn, b]
    else:
        xn = b
        intervalo = [a, xn]
    #x_anterior sera utilizado para calcular o DeltaXn
    x_anterior = None

    cont_dicotomia = 0
    cont_newton = 0

    #lista responsavle para guardar as aproximacoes
    historico = []
    #loop so continua enquanto nao bateu as iterações maximas
    n = 0
    while n < MAXIT:
        historico.append(float(xn))
        #=== abaixo faremos uma lista de checks para verificar se faz sentido continur a procurar===
        #primeria checagem consiste em ver se a raiz foi encontrada, caso sim, simplesmente retornamos que foi encontrada
        if f(xn) == 0:
            if verbose:
                print(f"Raiz encontrada e é {xn}, pois f({xn}) = 0")
            break

        #secundo check verificar se a condicao inicial é satisfeita
        if sinais_opostos(intervalo[0], intervalo[1], f) is not True:
            if verbose:
                print("Condição inicial f(a) * f(b) < 0 não é satisfeita")
            break

        #no nosso codigo caso estejamos na 1 iteracao (x_anterior = None) assumimos o delta_xn como o proprio xn
        #para evitar erros e nao multiplicarmos ou dividirmos por zero
        if x_anterior is not None:
            delta_xn = xn - x_anterior
            if sinais_quebra_1(delta_xn, xn, ATOL, RTOL):
                if verbose:
                    print(f"O número máximo de interações não foi atingido e a execução será encerrada pois a condição '|Δxn| < ATOL + RTOL*|xn|' foi satisfeita. \n")
                break
    
        else:
            delta_xn = xn

        #na primeira interacao nao temos x_anterior e portanto nem faz sentido falarmos do método de Newton
        #por essa razao a 1 iteracao sempre fazemos a dicotomia
        if n == 0: 
            x_anterior = xn
            xn = dicotomia(intervalo[0], intervalo[1])
            intervalo = atualiza_intervalo(intervalo[0], intervalo[1], f, xn)
            cont_dicotomia += 1
            n += 1
            continue 
        #===fim das checagens===

        #
        if checa_condicao2(intervalo[0], intervalo[1], f, f_linha, xn) and checa_condicao3(f, f_linha, xn, delta_xn):
            x_anterior = xn
            xn = newton(xn, f, f_linha)
            intervalo = atualiza_intervalo(intervalo[0], intervalo[1], f, xn)
            cont_newton += 1

        else:
            x_anterior = xn
            xn = dicotomia(intervalo[0], intervalo[1])
            intervalo = atualiza_intervalo(intervalo[0], intervalo[1], f, xn)
            cont_dicotomia += 1
        n += 1

    dic = {'Historico': historico,
           'Raiz': historico[-1], 
           'ContadorDicotomia' : cont_dicotomia, 
           'ContadorNewton' : cont_newton, 
           'NumeroIteracoes': n}
    
    return dic

#=== 3. MODO INTERATIVO (usa input e chama newton_modificado) ===
#funcao responsavel para interagir com o usuario apenas
def modo_interativo():
    while True:
        expressao_str = input("Digite f(x) em termos de 'x' (ou 'exit' para sair): ")
        if expressao_str.lower() == 'exit':
            break
        f, f_linha = calcula_derivada(expressao_str)

        if f is None or f_linha is None:
            print("Não foi possível interpretar a função digitada. Tente novamente \n")
            continue

        try:
            a = float(input("Entre com o primeiro valor do intervalo: "))
            print(f"Intervalo: [{a}, None]")
            b = float(input("Entre com o segundo valor do intervalo: "))
            print(f"Intervalo: [{a}, {b}]")
        except ValueError:
            print("Valor inválido para o intervalo. Tente novamente \n")
            continue

        if not sinais_opostos(a, b, f):
            print("Condição inicial f(a) * f(b) < 0 não satisfeita")
            continue

        try:
            ATOL = float(input("Entre com o valor para o erro absoluto (ATOL): "))
            RTOL = float(input("Entre com o valor para o erro relativo (RTOL): "))
            MAXIT = int(input("Entre com o valor para o número de interações: "))
        except ValueError:
            print("Valor inválido para os parâmetros. Tente novamente \n")
            continue

        resultado = newton_modificado(f, f_linha, a, b, ATOL, RTOL, MAXIT)
        print("----- -----\n")
        print(f"1) A raiz aproximada para sua expressão é: {resultado['Raiz']}")
        print(f"2) Foram necessárias {resultado['NumeroIteracoes']} interações para aproximar a raiz com base nos erros")
        print(f"3) O método da dicotomia foi usado {resultado['ContadorDicotomia']} vezes para aproximar a raiz")
        print(f"4) O método de Newton foi usado {resultado['ContadorNewton']} vezes para aproximar a raiz")
        print(f"5) O histórico de aproximações é tal que:\n{resultado['Historico']}")

# === 4. TESTES / APLICACOES ===
#Para cada teste os valores de ATOL, RTOL e MAXIT foram 10^-6 e 100, nosso objetivo foi minimizar o erro o máximo possível
#Já para os intervalos avaliamos individualmente onde cada função dos testes trocava de sinal, por exemplo, fomos
#verificando para quais valores a função do teste de queda trocava de sinal (Isso foi feito manualmente, a papel e lapis)
#ao encontrarmos aplicamos esses intervalos
def teste_queda_corpo():
    expressao = ('((10 - exp(-2*x) * (10 - 3*x)) - 20*x)/x')
    f, f_linha = calcula_derivada(expressao)
    intervalo = [0.1, 0.2] #para testes feitos a mao, descobrimos que a funcao troca de sinal entre 0.1 e 0.2
    resultado = newton_modificado(f, f_linha, intervalo[0], intervalo[1], 0.000001, 0.000001, 100)

    print("Teste: Queda do corpo (Encontrar k)")
    print(f"1) k aproximado: {resultado['Raiz']}")
    print(f"2) Foram necessárias {resultado['NumeroIteracoes']} iterações para aproximar a raiz com base nos erros")
    print(f"3) O método da dicotomia foi usado {resultado['ContadorDicotomia']} vezes para aproximar a raiz")
    print(f"4) O método de Newton foi usado {resultado['ContadorNewton']} vezes para aproximar a raiz")
    print(f"5) O histórico de aproximações é tal que:\n{resultado['Historico']}")

def teste_catenaria():
    expressao = "x*(cosh(10/x) - 1) - 0.5"
    f, f_linha = calcula_derivada(expressao)
    intervalo = [95, 105]

    resultado = newton_modificado(f, f_linha, intervalo[0], intervalo[1], 0.000001, 0.000001, 100)
    print("Teste: catenária (achar β)")
    print(f"1) β aproximado: {resultado['Raiz']}")
    print(f"2) Foram necessárias {resultado['NumeroIteracoes']} iterações para aproximar a raiz com base nos erros")
    print(f"3) O método da dicotomia foi usado {resultado['ContadorDicotomia']} vezes para aproximar a raiz")
    print(f"4) O método de Newton foi usado {resultado['ContadorNewton']} vezes para aproximar a raiz")
    print(f"5) O histórico de aproximações é tal que:\n{resultado['Historico']}")

# === Polinômios de Legendre ===
#aqui fazemos todos as funcoes necessarias para fazer o teste de gauss_legendre
def legendre(N, x):
    if N == 0:
        return 1.0
    P0, P1 = 1.0, x
    for k in range(1, N):
        P2 = ((2*k + 1) * x * P1 - k * P0) / (k + 1)
        P0, P1 = P1, P2
    return P1

def legendre_derivada(N, x):
    if N == 0:
        return 0.0
    PN = legendre(N, x)
    PN_1 = legendre(N - 1, x)
    return N * (x * PN - PN_1) / (x**2 - 1)


def gauss_legendre_todos(N_max, ATOL=1e-12, RTOL=1e-12, MAXIT=100):
    raizes_por_N = {1: [0.0]}
    pesos_por_N = {1: [2.0]}

    for n in range(2, N_max + 1):
        anteriores = sorted(raizes_por_N[n - 1])
        pontos = [-1.0] + anteriores + [1.0]

        f = lambda x, n=n: legendre(n, x)
        f_linha = lambda x, n=n: legendre_derivada(n, x)

        novas_raizes = []
        for i in range(len(pontos) - 1):
            a, b = pontos[i], pontos[i + 1]
            resultado = newton_modificado(f, f_linha, a, b, ATOL, RTOL, MAXIT, verbose=False)
            novas_raizes.append(resultado['Raiz'])

        raizes_por_N[n] = sorted(novas_raizes)
        pesos_por_N[n] = [2 / ((1 - xj**2) * legendre_derivada(n, xj)**2) for xj in raizes_por_N[n]]

    return raizes_por_N, pesos_por_N


def teste_gauss_legendre():
    raizes_por_N, pesos_por_N = gauss_legendre_todos(16)

    print("=== Teste: Quadratura de Gauss-Legendre (N=2 a 16) ===")
    for N in range(2, 17):
        raizes, pesos = raizes_por_N[N], pesos_por_N[N]
        print(f"N={N}")
        for xj, wj in zip(raizes, pesos):
            print(f"  x = {xj:.10f}   w = {wj:.10f}")
        print(f"  soma dos pesos = {sum(pesos):.10f}  (deve ser 2)\n")


if __name__ == "__main__":
    while True:
        entrada = input("Caso queira observar os resultados dos testes digite 'Queda', 'Catenaria' ou 'Gauss'.\n" \
        "Caso queira aproximar a raiz de uma funcao, digite 'Aproximar'.\n" \
        "Para sair digite 'exit'\n"
        ">>> ")
        if entrada.lower() == 'queda':
            teste_queda_corpo()
        elif entrada.lower() == 'catenaria':
            teste_catenaria()
        elif entrada.lower() == 'gauss':
            teste_gauss_legendre()
        elif entrada.lower() == 'aproximar':
            modo_interativo()
        elif entrada.lower() == 'exit':
            break
        else:
            print("Entrada inválida. Tente novamente\n")
            continue