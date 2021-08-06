

# Copyright (c) 2021 J Hudson.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import unittest
import sys
import os
import re

p=os.path.abspath( os.path.join( os.path.dirname(__file__), "..") )
sys.path.insert( 0, p )

from syparser import Parser
import syparser

def evaluate(n):
    if n.sym == "NUM":
        return float(n.token.lexeme)
    elif n.sym == "ADDOP" or n.sym == "PLUS" or n.sym == "ADD":
        v1 = evaluate(n.children[0])
        v2 = evaluate(n.children[1])
        if n.lexeme == "+":
            return v1+v2
        elif n.lexeme == "-":
            return v1-v2
        else:
            assert 0
    elif n.sym == "MULOP" or n.sym == "MUL":
        v1 = evaluate(n.children[0])
        v2 = evaluate(n.children[1])
        if n.lexeme == "*":
            return v1*v2
        elif n.lexeme == "/":
            return v1/v2
        else:
            assert 0
    elif n.sym == "POWOP":
        v1 = evaluate(n.children[0])
        v2 = evaluate(n.children[1])
        return v1**v2
    elif n.sym == "SUBOP":
        v1 = evaluate(n.children[0])
        v2 = evaluate(n.children[1])
        return v1-v2
    elif n.sym == "NEGATE":
        assert len(n.children) == 1
        return -evaluate(n.children[0])
    elif n.sym == "COMMA":
        v1 = evaluate(n.children[0])
        v2 = evaluate(n.children[1])
        if type(v1) != list:
            assert type(v1) == float
            v1 = [v1]
        if type(v2) != list:
            assert type(v2) == float
            v2 = [v2]
        return v1 + v2
    elif n.sym == "ABS":
        assert len(n.children) == 1
        return abs(evaluate(n.children[0]))
    elif n.sym == "FUNCCALL":
        assert n.children[0].lexeme == "sum"
        if n.children[1].sym == "NOARGS":
            return 0
        lst = evaluate(n.children[1])
        if type(lst) == float:
            return lst
        if len(lst) == 0:
            return 0
        return sum(lst)
    else:
        print("Unknown sym:",n.sym)
        assert 0

def treecheck(root,lst):

    try:
            
        #lst tells what should be in the tree
        assert isinstance(root,syparser.TreeNode)
        
        n = root
        p=[]
        for ctr,item in enumerate(lst):
            if item.startswith("left="):
                assert len(n.children) > 0
                rest = item[5:]
                assert n.children[0].sym == rest or n.children[0].lexeme==rest
                assert len(n.children[0].children) == 0
            elif item.startswith("right="):
                assert len(n.children) == 2
                rest = item[6:]
                assert n.children[1].sym == rest or n.children[1].lexeme==rest
                assert len(n.children[1].children) == 0
            elif item == "left":
                assert len(n.children) > 0
                p.append(n)
                n = n.children[0]
            elif item == "right":
                assert len(n.children) == 2
                p.append(n)
                n = n.children[1]
            elif item == "up":
                n = p.pop()
            else:
                assert n.sym == item or n.lexeme == item
        return 
    except AssertionError:
        print("Error:", " ".join(str(q) for q in lst[:ctr]),"--->",lst[ctr],"<---"," ".join([str(q) for q in lst[ctr+1:]]))
        print("Current node:",n)
        raise
    
    return
    
    i=0
    def walk(n):
        nonlocal i
        for c in n.children:
            assert isinstance(c,syparser.TreeNode)
            walk(c)
    
        if type(lst[i]) == str:
            ok = (n.lexeme == lst[i] or n.sym == lst[i])
        else:
            sym,tok = lst[i]
            ok = ( (sym==None or n.sym == sym) and (tok==None or tok.lexeme == n.lexeme))
        if not ok:
            print("\n\nError at",i)
            print(c.lexeme)
            print(lst)
            print(lst[i]==c.lexeme)
            print(" ".join(lst[:i]),"--->",lst[i],"<---"," ".join(lst[i+1:]) )
            assert 0
        i+=1
        
    walk(root)
    

class Test1(unittest.TestCase):
          
    def check(self,r,expected):
        actual = evaluate(r)
        self.assertEqual( actual, expected )
        
    def test_basic1(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        r = P.parse("42")
        self.check(r,42)

    def test_basic2(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        r = P.parse("42 + 5")
        self.check(r,47)

    def test_basic3(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        r = P.parse("42 - 5 + 7")
        self.check(r,44)
        
    def test_basic4(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        r = P.parse("2 + 3*4")
        self.check(r,14)

    def test_basic5(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        r = P.parse("2 + 3*4")
        self.check(r,14)

    def test_basic6(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        r = P.parse("8 / 2 / 16")
        self.check(r,0.25)
        
    def test_basic7(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        r = P.parse("3+4**2")
        self.check(r,19)
        
    def test_basic8(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        r = P.parse("3+4**2")
        self.check(r,19)
        
    def test_basic9(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        r = P.parse("2**4**3")
        self.check(r, 2**(4**3) )
        
    def test_basic10(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        r = P.parse("2*(3+4)")
        self.check(r, 14 )
        
    def test_basic11(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        r = P.parse("(3+4)*2")
        treecheck(r, ("*", "left", "+", "left", "3", "up", "right","4","up","up","right","2") )
        self.check(r, 14 )

    def test_basic12(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator(sym="ADDOP", regex="[-+]", precedence=10)
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addOperator(sym="MULOP", regex=re.compile("[*/]"), precedence=20)
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]",
                            addToTree=True)
        r = P.parse("(3+4)*2")
        treecheck(r, ("*", "left", "LP", "left", "+", "left", "3", "up", "right","4","up","up","up","right","2") )

    def negcheck(self,inp,expected,tree=None):
        P = syparser.Parser()
        P.addOperand(sym="NUM",regex="[0-9]+")
        P.addOperator(sym="ADDOP",regex="[+]",precedence=10)
        P.addOperator(sym="SUBOP",regex="-",precedence=10)
        P.addOperator(sym="MULOP",regex="[*/]",precedence=15)
        P.addNegationStyleOperator(sym="NEGATE", transformFromSym="SUBOP", precedence=20 )
        P.addOperator(sym="POWOP", regex=re.compile("[*]{2}"), precedence=30, leftAssociative=False)
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        root = P.parse(inp)
        self.check( root ,expected)
        if tree:
            treecheck(root,tree)
            
    def test_neg1(self):
        self.negcheck( ("-2") , -2)
    def test_neg2(self):
        self.negcheck( ("- 2") , -2)
    def test_neg3(self):
        self.negcheck( ("5 + -2") , 3)
    def test_neg4(self):
        self.negcheck( ("5 + - 2") , 3)
    def test_neg5(self):
        self.negcheck( ("5+-2") , 3)
    def test_neg6(self):
        self.negcheck( ("5--2") , 7)
    def test_neg7(self):
        self.negcheck( ("5 --2") , 7)
    def test_neg8(self):
        self.negcheck( ("5 - - 2") , 7)
    def test_neg9(self):
        self.negcheck( ("-2 + 5") , 3)
    def test_neg10(self):
        self.negcheck( ("(-2 + 5)") , 3)
    def test_neg11(self):
        self.negcheck( ("-4**2") , -16)      #powop is higher precedence
    def test_neg12(self):
        self.negcheck( ("-4*2") , -8)    
    def test_neg13(self):
        self.negcheck( "-4*2", -8, ( "*", "left", "NEGATE", "left", "4", "up", "up", "right", "2" ) )
    def test_neg14(self):
        self.negcheck( ("-5+-2") , -7)

    def test_implied0(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        treecheck( P.parse( "f(x)") , ("FUNCCALL","left","f","up","right","x") )


    def test_implied1(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        P.addImplicitOperand(sym="VOID", symbolBefore="LP", symbolAfter="RP")
        treecheck( P.parse( "f()") , ("FUNCCALL","left=f", "right=VOID"))
 
    def test_implied1a(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        P.addImplicitOperand(sym="VOID", symbolBefore="LP", symbolAfter="RP")
        treecheck( P.parse( "f(x)") , ("FUNCCALL","left","f","up","right","x") )


    def test_implied2(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        treecheck( P.parse( "f(2,4)") , ("FUNCCALL", "left=f","right","COMMA","left","2","up","right","4"))

    def test_implied3(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="MUL", symbolBefore="ID", symbolAfter="ID")
        P.addImplicitOperator( sym="MUL", symbolBefore="RP", symbolAfter="LP")
        treecheck( P.parse( "x y"),  ( "MUL", "left=x", "right=y" ) )
        treecheck( P.parse( "(x) (y)"),  ("MUL","left=x","right=y") )
        
    def test_funccall1(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"],symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        self.check( P.parse( "1+sum(2)"), 3 ) 
        self.check( P.parse( "1+sum(2,3)"), 6 ) 
        self.check( P.parse( "1+sum(2,3,4)"), 10 ) 
        self.check( P.parse( "1+sum()"), 1 ) 
        treecheck( P.parse("1+sum()"), ("+","left=1","right","FUNCCALL", "left=sum", "right=NOARGS" ) )

    def test_funccall2(self):
        P = syparser.Parser()
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]",
                             permitEmpty=True )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        self.check( P.parse( "1+sum(2)" ), 3 ) 
        self.check( P.parse( "1+sum(2,3)"), 6 ) 
        self.check( P.parse( "1+sum(2,3,4)"), 10 ) 
        with self.assertRaises(syparser.MissingOperandException):
            self.check( P.parse( "1+sum()"), 1  )
        
    # ~ def test_absval(self):
        # ~ P = syparser.Parser()
        # ~ P.addOperator("PLUS","[+]",precedence=10)
        # ~ P.addOperator("MUL","[*]",precedence=20)
        # ~ P.addOperand(sym="NUM", regex="-?\\d+")
        # ~ P.addGroupingSymbol( openingSymbol="ABS", openingRegex=r"\|", 
                             # ~ closingSymbol="ABS", closingRegex=r"\|",
                             # ~ addToTree=True )
        # ~ P.addGroupingSymbol( openingSymbol="LP", openingRegex=r"\(", 
                             # ~ closingSymbol="RP", closingRegex=r"\)" )
                             
        # ~ self.check( P.parse("3*|-4+2|"), 6)
        # ~ self.check( P.parse("3*(-4+2)"), -6)
        # ~ print(Parser.toText(P.parse("1 + | |2+3| + |4+5| |")))

                    
    def test_ambiguousmatch(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM1", regex="\\d+" )
        P.addOperand( sym="NUM2", regex="\\w+" )
        with self.assertRaises(syparser.AmbiguousMatchException):
            P.parse("42")
 
    def test_zerolengthmatch(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d*" )
        with self.assertRaises(syparser.ZeroLengthMatchException):
            P.parse("x")

    def test_nomatch(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        with self.assertRaises(syparser.NoTokenMatchException):
            P.parse("x")

    def test_missingoperator(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        with self.assertRaises(syparser.MissingOperatorException):
            P.parse("1 + 12 10")
            
    def test_missingoperator2(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        with self.assertRaises(syparser.MissingOperatorException):
            P.parse("12 10")
            
    def test_missingoperand(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        with self.assertRaises(syparser.MissingOperandException):
            P.parse("12 +")

    def test_missingoperand2(self):
        #we have two terminals with the same length of match
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )

        with self.assertRaises(syparser.MissingOperandException):
            P.parse(" (12 + ) + 4")

    def test_empty1(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        with self.assertRaises(syparser.EmptyInputException):
            P.parse("")
        
    def test_empty2(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        P.addComment( r"/\*.*?\*/" )
        with self.assertRaises(syparser.EmptyInputException):
            P.parse("/* foo */")
       
    def test_mismatchgroup(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        P.addGroupingSymbol(openingSymbol="LB", openingRegex="\[",
                            closingSymbol="RB", closingRegex="\]" )
        with self.assertRaises(syparser.MismatchedGroupingException):
            P.parse("(1+2]")
            
    def test_unclosedgroup(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        P.addGroupingSymbol(openingSymbol="LB", openingRegex="\[",
                            closingSymbol="RB", closingRegex="\]" )
        with self.assertRaises(syparser.UnclosedGroupingException):
            P.parse("(1+2")
            
    def test_unexpectedGrouping(self):
        P = Parser()
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addOperand( sym="NUM", regex="[a-z]+" )
        P.addOperator( sym="ADDOP", regex="[+]", precedence=10 )
        P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
                            closingSymbol="RP", closingRegex="[)]" )
        P.addGroupingSymbol(openingSymbol="LB", openingRegex="\[",
                            closingSymbol="RB", closingRegex="\]",
                            openingSymbolMustFollow="ID")
        with self.assertRaises(syparser.UnexpectedGroupingException):
            P.parse("1+2[3]")
        
    def test_emptyparens(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
            symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        with self.assertRaises(syparser.EmptyGroupException):
            P.parse("1 + () 2")
            
    def test_emptyparens2(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
            symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        with self.assertRaises(syparser.EmptyGroupException):
            P.parse("1 + ()")
            
    def test_emptyparens3(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
            symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]",
                             addToTree=True )
        with self.assertRaises(syparser.EmptyGroupException):
            P.parse("1 + ()")
 

    def test_emptyparens4(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
            symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]",
                             addToTree=True,
                             permitEmpty=True)
        
        with self.assertRaises(syparser.MissingOperandException):
            treecheck( P.parse("1 + ()"),  ( "+", "left=1", "right=LP" ) )
 
    def test_emptyparens5(self):
        P = syparser.Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperand(sym="NUM", regex="[0-9]+")
        P.addOperator(sym="ADD", regex="[+]", precedence=10)
        P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
        P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
            symbolAfter="RP")
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]",
                             permitEmpty=True)
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")

        treecheck( P.parse("1 + () 2"), ( "+", "left=1", "right=2") )
        
  
if __name__=="__main__":
    unittest.main()
