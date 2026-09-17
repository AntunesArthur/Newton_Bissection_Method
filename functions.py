import numpy as np
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
def sinais_opostos(a, b, func):
    if func(a) * func(b) < 0: return True
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
def newton(a, b, func):
    p1 = a - (func(a)/func(a))