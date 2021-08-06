
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


#https://www.sphinx-doc.org/en/master/usage/restructuredtext/domains.html
#https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html

"""
This module contains a simple parser for mathematical expressions. 
It's pretty flexible and can be used for a lot more than that.


Basic example::

    P = syparser.Parser()
    P.addOperator(sym="ADD", precedence=10, regex="[-+]")
    P.addOperator(sym="MUL", precedence=20, regex="[*/]")
    P.addOperator(sym="POW", precedence=30, regex="[*]{2}", leftAssociative=False)
    P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
        closingSymbol="RP", closingRegex="[)]")
    P.addOperand(sym="NUM", regex="[0-9]+")


Now that the parser has been created, it can be used to parse strings::

    root = P.parse("(1+2)*3**4")


This returns the root of a parse tree. For this example,
the tree looks like this::


                  MUL(*)
                 /      \\
           ADD(+)        POW(**)
          /   \\        /     \\
        NUM(1) NUM(2)  NUM(3) NUM(4)
        

"""

import sys
import re
import typing
from typing import List, Union, Pattern, Optional, TextIO, Generic, TypeVar
import logging

_log = logging.getLogger(__name__)

if 0:
    logging.basicConfig(level=logging.DEBUG)


class ParseException(Exception):
    """Generic exception thrown when there is an error in parsing.

    Args:
        message (str): The exception message.
        inputString (str): The input that was being parsed.
    """
    
    def __init__(self,message: str,*args):
        super().__init__(message)
        self.message = message        
        
    def __repr__(self):
        return str(self)

class EmptyInputException(ParseException):
    """Exception raised when the input is empty."""
    pass
    


class TokenizeException(ParseException):
    """Generic exception for errors when tokenizing."""
    
    def __init__(self,*, message: str, inputIndex: int, line: int, column: int):
        super().__init__(message=message)
        self.inputIndex=inputIndex      #: Character index where the error occurred. Indexing starts at 0.
        self.line = line                #: Line where the error occurred. Line numbering starts at 1.
        self.col = column               #: Column where the error occurred. Column numbering starts at 0.
    def __str__(self):
        return f"{self.message}: At line {self.line}, col {self.col}, index {self.inputIndex}"


class EmptyGroupException(TokenizeException):
    """Exception raised when a grouping is empty and permitEmpty is False."""
    pass
    
    
class UnexpectedGroupingException(TokenizeException):
    """Exception produced when an invalid grouping symbol is encountered.
    See :meth:`addGroupingSymbol` for more details."""
    
    def __init__(self,*,message:str, inputIndex: int, sym: str, line: int, column: int):
        super().__init__(message=message, inputIndex=inputIndex, line=line, column=column)
        self.sym=sym                    #: The symbol for the invalid grouping symbol

class ZeroLengthMatchException(TokenizeException):
    """Exception raised if a regular expression matches the input but the length of
        the match is zero.
        This typically indicates a problem with the regular expression: Zero length matches
        should not be permitted. The tokenizer will try to find a longer match if possible;
        this exception is only raised if no longer match exists.
    """
    def __init__(self, *, message: str, inputIndex: int, sym: str, line:int, column: int):
        super().__init__(message=message, inputIndex=inputIndex, line=line, column=column)
        self.sym=sym                    #: The symbol that had the zero length match
  
class NoTokenMatchException(TokenizeException):
    """Exception raised when the input doesn't match any possible token."""
    pass

class AmbiguousMatchException(TokenizeException):
    """Exception raised when more than one possible match at a given input location.
        Note that matches are only ambiguous if they are the same length;
        the tokenizer always tries to take the longest match. This exception
        is only raised if there are two or more matches of the same length
        and there are no matches longer than that.
    """
    def __init__(self, *, message: str, inputIndex: int, line:int, column: int, sym1: str, sym2: str):
        super().__init__(message=message, inputIndex=inputIndex,line=line, column=column)
        self.sym1 = sym1        #: First matching symbol name
        self.sym2 = sym2        #: Second matching symbol name
        
    def __str__(self):
        return "{}: {} {}".format(self.message, self.sym1, self.sym2)
    def __repr__(self):
        return str(self)
    
class TreeBuildingException(ParseException):
    """Generic exception for errors when building tree"""
    def __init__(self,*, message: str, errorToken: "Token"):
        super().__init__(message=message)
        self.errorToken = errorToken        #: Token where the error was noticed
    
    def __str__(self):
        return f"{self.message}: {self.errorToken}"

class MissingOperatorException(TreeBuildingException):
    """
    Exception raised when an operator is missing. Example::
    
        1 + 2 3
        
    """
    def __init__(self,*, message: str, errorToken: "Token"):
        super().__init__(message=message, errorToken=errorToken)
 
    
class MissingOperandException(TreeBuildingException):
    """
    Exception raised when an operand is missing. Example::
    
            2 +
        
    """
    def __init__(self,*, message: str, errorToken: "Token"):
        super().__init__(message=message, errorToken=errorToken)

class MismatchedGroupingException(TreeBuildingException):
    """
    Exception raised when mismatched grouping symbols are seen. For example::
        
        1 - ( 2 + 3 ]
    """
      
    def __init__(self,*, message: str, errorToken: "Token", mismatchedToken: "Token"):
        super().__init__(message=message, errorToken=errorToken)
        self.mismatchedToken = mismatchedToken      #: The opening grouping symbol that was mismatched


class UnclosedGroupingException(TreeBuildingException):
    """
    Exception raised when an unclosed grouping symbol is seen. For example::
        
        1 - ( 2 + 3 
    """
    
    def __init__(self,message:str, errorToken: "Token", mismatchedToken: "Token"):
        super().__init__(message=message, errorToken=errorToken)
        self.mismatchedToken = mismatchedToken
                                         
class Token:
    """A single token."""
    
    
    def __init__(self, terminal: "_Terminal", lexeme: Optional[str], line: int, column: int, index: int):
        self.terminal = terminal          #Not really useful outside of the parser
        self.lexeme: str = typing.cast(str,lexeme)         #: The lexeme (text) of this token
        self.line: int = line             #: Input line number, starting from 1, where this token occurs.
        self.col: int = column            #: Column number, starting from 0, where the token occurs.
        self.index: int = index           #: Character index, starting from 0, within the input where this token starts.

    sym = property(lambda self: self.terminal.sym)      #: The symbol of this token
    
    def __str__(self) -> str:
        return f"[Token: {self.sym} {self.line} {self.lexeme}]"
         
    def __repr__(self) -> str:
        return str(self)
    

class TreeNode:
    """A single node in the parse tree."""
    
    ctr=0
    def __init__(self,token: Token):
        self.token : Token =token    #: Token for this node
        self.children: List[TreeNode] = []     #: List of child TreeNode objects
        self.unique = f"n{TreeNode.ctr}"
        TreeNode.ctr+=1
    def addChild(self,c):
        assert isinstance(c,TreeNode)
        self.children.append(c)
        
    sym = property(lambda self: self.token.sym)     #: Symbol for this node.
    lexeme = property(lambda self: self.token.lexeme)     #: Lexeme (text) for this node.
    
    def __str__(self) -> str:
        return f"[TreeNode: {self.sym} {self.lexeme} with {len(self.children)} {'children' if self.children!=1 else 'child'}]"
    def __repr__(self) -> str:
        return str(self)
      
_T = TypeVar("_T")
class _Stack(Generic[_T]):
    def __init__(self,label: str):
        self.L: List[_T] = [] 
        self.label=label
    def push(self,x: _T):
        _log.debug("%s: Push %s",self.label,x)
        self.L.append(x) 
    def pop(self) -> _T:
        x = self.L.pop() 
        _log.debug("%s: Pop %s", self.label,x )
        return x
    def empty(self) -> bool:
        return len(self.L) == 0
    def top(self) -> _T:
        return self.L[-1]
    def size(self):
        return len(self.L)
    def __str__(self):
        return str(self.L)
    def __repr__(self):
        return str(self)
        
def _makeRegex(regex: Union[str,Pattern]  ) -> Pattern:
    if type(regex) == str:
        regex = re.compile(regex,re.IGNORECASE)
    return typing.cast(typing.Pattern,regex)
        
#all terminals are singletons
class _Terminal:

    sym: str
    regex: Optional[Pattern]
    
    def __init__(self,*,sym: str, regex : Union[str,Pattern,None]=None):
        self.sym: str = sym   
        if type(regex) == str:
            self.regex = re.compile(typing.cast(str,regex),re.IGNORECASE)
        elif regex == None:
            self.regex = None
        else:
            self.regex = typing.cast(Pattern,regex)
            
    def isOperator(self) -> bool:
        return False
    def isOperand(self) -> bool:
        return False
    def omitWhenTokenizing(self) -> bool:
        return False
    def isGrouping(self) -> bool:
        return False
    def opensGroup(self) -> bool:
        return False
    def closesGroup(self) -> bool:
        return False   
    def matchesTerminal(self,t: "_Terminal"):
        return False
    def mustFollow(self):
        return None
    def addToTree(self):
        return True
    def __str__(self) -> str:
        return f"[_Terminal: {self.sym} {str(self.regex)}]"
    def __repr__(self) -> str:
        return str(self)
        
class _CommentTerminal(_Terminal):
    def __init__(self,*,sym:str,regex:Union[str,Pattern]):
        super().__init__(sym=sym,regex=regex)
    def omitWhenTokenizing(self) -> bool:
        return True
        
class _WhitespaceTerminal(_Terminal):
    def __init__(self,*,sym:str,regex:Union[str,Pattern]):
        super().__init__(sym=sym,regex=regex)
    def omitWhenTokenizing(self) -> bool:
        return True
     
class _GroupingTerminal(_Terminal):
    def __init__(self,*,sym:str,regex: Union[str,Pattern], addToTree: bool, permitEmpty: bool):
        super().__init__(sym=sym,regex=regex)
        self.addToTree_ = addToTree
        self.permitEmpty=permitEmpty
    def isGrouping(self) -> bool:
        return True
    def addToTree(self):
        return self.addToTree_
    
class _OpeningGroupingTerminal(_GroupingTerminal):
    def __init__(self,*,sym:str,regex: Union[str,Pattern], mustFollow: Union[str,List[str],None], addToTree: bool,
                permitEmpty: bool):
        _GroupingTerminal.__init__(self,sym=sym,regex=regex, addToTree=addToTree, permitEmpty=permitEmpty)
        
        if mustFollow == None:
            lst=None
        elif type(mustFollow) == str:
            lst = [mustFollow]
        else:
            lst = typing.cast(List,mustFollow)[:]
        self.mustFollow_=lst
    def mustFollow(self):
        return self.mustFollow_
    def opensGroup(self) -> bool:
        return True
        
class _ClosingGroupingTerminal(_GroupingTerminal):
    def __init__(self,*,sym:str,regex: Union[str,Pattern], openingTerminal: _Terminal, addToTree: bool, permitEmpty: bool):
        super().__init__(sym=sym,regex=regex,addToTree=addToTree,permitEmpty=permitEmpty)
        self.openingTerminal = openingTerminal
    def closesGroup(self) -> bool:
        return True
    def matchesTerminal(self,t: _Terminal):
        #Returns true if t is the terminal that matches this terminal.
        #Ex: If this terminal is a right paren, isMatchingTerminal
        #would return true for the left paren
        return t == self.openingTerminal
    
# ~ class _OpeningAndClosingGroupingTerminal(_OpeningGroupingTerminal,_ClosingGroupingTerminal):
    # ~ def __init__(self,*,sym:str,regex: Union[str,Pattern], 
            # ~ mustFollow: Union[str,List[str],None], addToTree: bool):
        # ~ _OpeningGroupingTerminal.__init__(self,sym=sym,regex=regex,mustFollow=mustFollow,addToTree=addToTree)
        # ~ _ClosingGroupingTerminal.__init__(self,sym=sym,regex=regex,openingTerminal=self,addToTree=addToTree)
        # ~ pass
                
                
class _OperatorTerminal(_Terminal):
    def __init__(self,*,sym: str,regex: Union[Pattern,None],leftAssociative: bool,arity: int,precedence: int):
        super().__init__(sym=sym,regex=regex)
        self.precedence=precedence
        self.arity=arity
        self.leftAssociative=leftAssociative
    def isOperator(self) -> bool:
        return True

class _NegationOperatorTerminal(_OperatorTerminal):
    def __init__(self,*,sym,leftAssociative,arity,precedence,transformFromSym):
        super().__init__(sym=sym,regex=None,leftAssociative=leftAssociative,precedence=precedence,arity=1)
        self.transformFromSym = transformFromSym
        
class _OperandTerminal(_Terminal):
    def __init__(self,*,sym: str,regex: Optional[Pattern]):
        super().__init__(sym=sym,regex=regex)
    def isOperand(self) -> bool:
        return True

class _ImplicitOperatorTerminal(_OperatorTerminal):
    def __init__(self,*, sym:str, precedence:int=sys.maxsize,
                    symbolBefore: Union[str,List[str]],symbolAfter: str):
        super().__init__(sym=sym,regex=None,precedence=precedence,arity=2,leftAssociative=True)
        if type(symbolBefore) == str:
            lst = typing.cast(List[str],[symbolBefore])
        else:
            lst = typing.cast(List[str],symbolBefore[:])
        self.symbolBefore=lst
        self.symbolAfter=symbolAfter
    
class _ImplicitOperandTerminal(_OperandTerminal):
    def __init__(self,*, sym:str,
                    symbolBefore: Union[str,List[str]],symbolAfter: str):
        super().__init__(sym=sym,regex=None)
        if type(symbolBefore) == str:
            lst = typing.cast(List[str],[symbolBefore])
        else:
            lst = typing.cast(List[str],symbolBefore)[:]
        self.symbolBefore=lst
        self.symbolAfter=symbolAfter
 
class Parser:
    """The parser class.
    
        Args:
        
            None
    """
    
    def __init__(self):
        """Create the parser
        """

        self.terminals = [
             _WhitespaceTerminal( sym="WHITESPACE", regex=re.compile(r"\s+"))
        ] 

        self.implicitEntities=[]
        self.negationStyleOperators=[]
 
    def tokenize(self,inp:str) -> List[Token]:
        """Tokenize the input. This is called automatically by the :meth:`parse` function.
        You don't need to call this method unless you just want a list of tokens without
        building the tree.
        
        Args:
            inp (str): String giving the input.
        
        Returns:
            List of :class:`Token` objects.
        """
        
        _log.debug("=================STARTING TOKENIZE====================")
        _log.debug("INPUT: %s",inp)
        
        terminals = self.terminals 
        
        idx = 0 
        tokens: List[Token] = [] 
        line = 1 
        
        inp = inp.replace("\r\n","\n") 
        inp = inp.replace("\r","\n") 
        
        while( idx < len(inp)):
            col = inp.rfind("\n",idx)
            if col == -1:
                col = idx
            else:
                col = idx - col
            
            matches=[]
            
            #keep the longest match
            for terminal in terminals:
                regex = terminal.regex 
                M = regex.match(inp,idx)
                
                if( M ):
                    matches.append( (M,terminal) )
            
            if len(matches) == 0:
                raise NoTokenMatchException(
                    message="Could not match a token",
                    inputIndex=idx,line=line,column=col) 
            
            matches.sort( key = lambda q: -len(q[0].group(0)) )
            if len(matches) > 1:
                m0,t0 = matches[0]
                m1,t1 = matches[1]
                if len(m0.group(0)) == len(m1.group(0)):
                    raise AmbiguousMatchException(message="Ambiguous token match",
                        sym1=t0.sym, sym2=t1.sym,
                        inputIndex=idx,line=line,column=col)
     
            bestMatch, terminal = matches[0]
            lexeme = bestMatch.group(0)
            
            if len(bestMatch.group(0)) == 0:
                raise ZeroLengthMatchException(message="A zero length match was the best one",
                    sym=terminal.sym,
                    inputIndex=idx,line=line,column=col) 
                    
            for t in self.negationStyleOperators:
                if terminal.sym == t.transformFromSym:
                    if( len(tokens) == 0 or (tokens[-1].terminal.isOperator() or tokens[-1].terminal.isGrouping() )):
                        terminal = t
                        break

            for fterminal in self.implicitEntities:
                list1 = fterminal.symbolBefore
                list2 = [q.sym for q in tokens[-len(list1):]]
                if list1 == list2 and terminal.sym == fterminal.symbolAfter:
                    tokens.append( Token(terminal=fterminal,
                                        lexeme=None,
                                        line=line, column=col, index=idx))
                    pterm = fterminal
                    break

            if terminal.opensGroup():
                if terminal.mustFollow() != None:
                    list1 = terminal.mustFollow()
                    list2 = tokens[-len(list1):]
                    if list1 != list2:
                        raise UnexpectedGroupingException(
                            message="This grouping symbol cannot appear here",
                            line=line, column=col,inputIndex=idx,sym=terminal.sym)
                            
            if terminal.closesGroup():
                if len(tokens) > 1 and terminal.matchesTerminal(tokens[-1].terminal):
                    if not terminal.permitEmpty:
                        raise EmptyGroupException(
                            message="Empty group",
                            line=line, column=col, inputIndex=idx )
                            
            if not terminal.omitWhenTokenizing():
                tokens.append(  Token( terminal=terminal, lexeme=lexeme, line=line, column=col, index=idx ) ) 
                pterm = terminal 


            line += lexeme.count("\n")
            idx += len(lexeme)
        
        _log.debug("Tokens: %s",tokens)
        
        return tokens 

    def makeTree(self,tokens: List[Token]) -> TreeNode:
        """
            Make the parse tree.
            Note: This function is called automatically by the parse() function.
            It does not normally need to be called directly.

            Args:
                tokens (list of :class:`Token`): The tokens
                inputString (str): The input string. Only used for error reporting.
            Returns:
                :class:`TreeNode`: The root of the parse tree
            
        """
        
        pendingOperators =  _Stack[Token]("operator") 
        pendingOperands = _Stack[TreeNode]("operand") 
        
        def applyOperator():

            
            optoken = pendingOperators.pop() 
            op =  typing.cast( _OperatorTerminal, optoken.terminal )

            _log.debug("applyOperator: %s",optoken)

            if( op.isGrouping() and op.opensGroup() ):
                raise UnclosedGroupingException(
                    message="Unclosed grouping symbol",
                    mismatchedToken=optoken, errorToken=t)
      
            if( op.arity == 2 ):
                if pendingOperands.size() < 2:
                    raise MissingOperandException(
                        message="Missing operand",
                        errorToken=optoken)
                rh = pendingOperands.pop() 
                lh = pendingOperands.pop() 
                n =  TreeNode(optoken)
                n.addChild(lh) 
                n.addChild(rh) 
                pendingOperands.push(n) 
            elif( op.arity == 1 ):
                if pendingOperands.size() < 1:
                    raise MissingOperandException(
                        message="Missing operand",
                        errorToken=optoken)
                lh = pendingOperands.pop() 
                n =  TreeNode(optoken)
                n.addChild(lh) 
                pendingOperands.push(n) 
            else:
                #should never happen
                assert 0
        
        # ~ groupNesting=[]
        
        for tokenIndex,t in enumerate(tokens):
            
            _log.debug("----------------")
            _log.debug("Token: %s",t)
            # ~ _log.debug("groupNesting: %s",groupNesting)
            
            
            terminal = t.terminal
            if terminal.isGrouping():
                # ~ if terminal.opensGroup() and terminal.closesGroup():
                    # ~ if len(groupNesting) > 0 and groupNesting[-1] == terminal:
                        # ~ _log.debug("opens and *closes* group")
                        # ~ opens=False
                    # ~ else:
                        # ~ _log.debug("*opens* and closes group")
                        # ~ opens=True
                if terminal.opensGroup():
                    _log.debug("opens group")
                    opens=True
                else:
                    _log.debug("closes group")
                    opens=False
                    
                if opens:
                    pendingOperators.push(t) 
                    # ~ groupNesting.append(terminal)
                else:
                    assert terminal.closesGroup()
                    # ~ groupNesting.pop()
                    
                    while True:
                        
                        if pendingOperators.empty():
                            raise MissingOperatorException(message="Missing operator",
                                    errorToken=t )
                                    
                        top = pendingOperators.top()
                        
                        if  terminal.matchesTerminal(top.terminal):
                            break
                            
                        if top.terminal.isGrouping():
                            assert top.terminal.opensGroup()
                            raise MismatchedGroupingException(message="Mismatched grouping symbols",
                                    mismatchedToken=pendingOperators.top(),
                                    errorToken = t )
                                    
                        applyOperator() 
                       
                    #endwhile
                    if terminal.addToTree():
                       tn = TreeNode(pendingOperators.pop())
                       if pendingOperands.size() < 1:
                           raise MissingOperandException(
                               message="Missing operand",
                               errorToken=tn.token)
                       tn.addChild(pendingOperands.pop())
                       pendingOperands.push(tn)
                    else:
                        pendingOperators.pop()  #discard opening grouping symbol
                    
            elif( t.terminal.isOperand() ):
                pendingOperands.push( TreeNode(t) )
            elif( t.terminal.isOperator() ):
                
                t_op = typing.cast(_OperatorTerminal, t.terminal)
                
                #if it's right associative, we always must let it wait for pending
                #operands...
                if t_op.leftAssociative:
                    while(True):
                    
                        if pendingOperators.empty():
                            break 
                    
                        p = pendingOperators.top() 
                        
                        p_op = typing.cast( _OperatorTerminal, p.terminal )
                        
                        if p_op.isGrouping() and p_op.opensGroup():
                            break 
                        elif( p_op.leftAssociative  and  p_op.precedence >= t_op.precedence ):
                            applyOperator() 
                        elif( not p_op.leftAssociative  and  p_op.precedence > t_op.precedence ):
                            applyOperator() 
                        else:
                            break   
                        
                pendingOperators.push(t)
                 
            else:
                #internal logic error
                assert 0

        
        while not pendingOperators.empty() :
            applyOperator() 
           
        _log.debug("End of parse: pendingOperands: %s",pendingOperands)
        
        root = pendingOperands.pop() 
        if( not pendingOperands.empty() ):
            opnd = pendingOperands.top()

            raise MissingOperatorException(message="Missing operator", errorToken=opnd.token)

        return root 
        

    
    def parse(self,inputString:str) -> TreeNode:
        """
        Parse the given string and return the root of the parse tree.
        This method does the following:
            
            * Call :math:`tokenize` to tokenize the input
            * Verify the list of tokens is not empty
            * Call :meth:`makeTree` to build the tree
            * Returns the root of the tree
        
        Args:
            inputString (str): The input to be parsed
        
        Returns:
            The root of the parse tree (a :class:`TreeNode` object).
            
        Raises:
            EmptyInputException if the input is empty.
        """
     
        t = self.tokenize(inputString) 
        if len(t) == 0:
            raise EmptyInputException(message="Empty input")
            
        root = self.makeTree(t) 
        return root 
           
           

    def addGroupingSymbol(self, *, 
                openingSymbol: str, 
                openingRegex: Union[str,Pattern], 
                closingSymbol: str, 
                closingRegex: Union[str,Pattern],
                openingSymbolMustFollow: str = None,
                permitEmpty = False,
                addToTree: bool = False ) -> None:
        """
            Add a  grouping symbol. The opening and closing symbols must be different.  
            
            For example, to allow the parser to recognize parentheses,
            you could do::
            
                P = syparser.Parser()
                P.addGroupingSymbol( openingSymbol="LPAREN", openingRegex="[(]", 
                                     closingSymbol="RPAREN", closingRegex="[)]" )
                                     
            It's possible to prevent grouping symbols from appearing "standalone."
            For example, this will allow brackets as array indexers (ex: A[i]) but not
            as standalone symbols (ex: 1 + [2*3] )::
            
                P = syparser.Parser()
                P.addOperand("ID","[a-z]+")
                P.addOperand("NUM", "\\d+")
                P.addImplicitOperator(sym="ARRAY", precedence=100, 
                    symbolBefore="ID", symbolAfter="LB")
                P.addGroupingSymbol( openingSymbol="LB", openingRegex="\\[", 
                                     closingSymbol="RB", closingRegex="\\]",
                                     openingSymbolMustFollow="ARRAY" )
                            
            For this example, the input "A[4]" would produce the tokens::
                
                ID(A), ARRAY, LB, NUM(4), RB
                
            And then this parse tree::
            
                    ARRAY
                    /   \\
                ID(A)  NUM(4)  
                   
            
            Args:
                openingSymbol (str): The name for the opening symbol
                openingRegex (string|regex): Regular expression for the opening symbol.
                                            If a string is provided,
                                            it is interpreted as a regular expression.
                closingSymbol (str): The name for the opening symbol
                closingRegex (string|regex): Regular expression for the opening symbol.
                                            If a string is provided,
                                            it is interpreted as a regular expression.
                openingSymbolMustFollow (str|None): If this is not None,
                                        it is taken as the name of a symbol that must
                                        immediately precede the opening symbol.
                                        If the opening grouping token is found in the input
                                        and it does *not* follow this symbol,
                                        an UnexpectedGroupingException
                                        will be raised.
                permitEmpty (bool): If an empty group is found (that is, the closing symbol
                                    immediately follows the opening symbol, like "()",
                                    this controls whether that's an error.
                                    If this is True, the syntax is allowed (although
                                    it might raise a parsing error for other reasons).
                                    If this is False, an empty group results in an
                                    EmptyGroupException being raised.
                addToTree (bool):  If this is True, the opening grouping symbol will have
                                   a node in the tree as the parent of the nodes
                                   so grouped.
     """
     
        opening = _makeRegex(openingRegex) 
        closing = _makeRegex(closingRegex)
        if openingSymbol == closingSymbol:
            raise RuntimeError("Opening and closing symbols must be different")
        ot =  _OpeningGroupingTerminal(sym=openingSymbol, regex=opening, mustFollow=openingSymbolMustFollow, addToTree=addToTree,permitEmpty=permitEmpty)
        ct =  _ClosingGroupingTerminal(sym=closingSymbol, regex=closing, openingTerminal=ot, addToTree=addToTree, permitEmpty=permitEmpty)
        self.terminals.append(ot)
        self.terminals.append(ct)
     
    def addOperand(self,*,sym:str,regex:Union[str,Pattern]) -> None:
        r"""
            Add an operand to the parser's language.
            Typical operands would be numbers or variables.
            
            For example::
            
                P = syparser.Parser()
                P.addOperand( sym="NUM", regex=r"\d+" )
                     
            Args:
                sym (str): The symbol name
                regex (str|regex): The regular expression for the symbol.
                                If a string is provided, it is interpreted 
                                as a regular expression. 
        """

        regex = _makeRegex(regex) 
        self.terminals.append(_OperandTerminal(sym=sym,regex=regex))


    def addOperator(self, sym:str, regex:Union[str,Pattern],
                    precedence:int, arity:int=2, leftAssociative:bool=True) -> None:
        """
        Add a  operator to the parser's language. For example::
    
            P = syparser.Parser()
            P.addOperator( sym="ADDOP", regex="[-+]", precedence=10, arity=2, leftAssociative=True)
            P.addOperator( sym="MULOP", regex="[*/%]", precedence=20, arity=2, leftAssociative=True)
            P.addOperator( sym="POWOP", regex="[*]{2}", precedence=30, arity=2, leftAssociative=False)
            P.addOperator( sym="BITNOT", regex="~", precedence=40, arity=1, leftAssociative=False)
        
        Args:
            sym (str): The operator's name
            regex (str|regex): The regular expression for the symbol.
                               If a string is provided, it is interpreted 
                               as a regular expression. 
            precedence (int): The precedence. Higher values = higher priority.
            arity (int): The only useful values here are 1 for a unary operator
                         or 2 for a binary operator.
            leftAssociative (bool): True if the operator is left associative; False if it's
                                    right associative.
        """
        if arity != 2 and arity != 1:
            raise RuntimeError("Arity must be 1 or 2")
            
        self.terminals.append(
            _OperatorTerminal(sym=sym,regex=_makeRegex(regex),
                leftAssociative=leftAssociative, precedence=precedence, arity=arity))

    def addNegationStyleOperator(self,*,sym:str,transformFromSym:str,precedence:int,leftAssociative:bool=False) -> None:
        """ Add a negation style operator. The most familiar case where
        this would be used is to allow the "-" sign to be used for both
        subtraction and negation. Example::
            
            P = syparser.Parser()
            P.addOperator(sym="SUB",regex="-",precedence=10)
            P.addNegationStyleOperator(sym="NEGATE", transformFromSym="SUB", precedence=20 )
            
        If a SUB token is found it will be transformed into a NEGATE token if any of these are true:
        
            * The SUB is at the beginning of the input
            * The SUB immediately follows an opening grouping symbol (ex: parentheses)
            * The SUB immediately follows an operator
            
        Args:
            sym (str): The new operator's name
            transformFromSym (str): The symbol that will be transformed into the new operator
            precedence (str): The precedence of the new operator
            leftAssociative (bool): True if the new operator is left associative
        """
        t = _NegationOperatorTerminal(
            sym=sym, precedence=precedence, arity=1,leftAssociative=leftAssociative,
            transformFromSym=transformFromSym)
        self.negationStyleOperators.append(t)
        
    def addImplicitOperator(self,*,sym:str,precedence:int=99999,symbolBefore:Union[str,List[str]],
                            symbolAfter:str,leftAssociative:bool=True) -> None:
        """
        Add an implicit operator. This is useful for things like function calls and
        array accesses that are not written explicitly in the input but nevertheless
        need nodes in the parse tree. The implicit operator will be created and its
        left and right children will be appended appropriately. Implicit operators are
        always binary. Example::
            
                #function calls, but not for zero-argument functions
                P = syparser.Parser()
                P.addOperand(sym="ID", regex="[a-z]+")
                P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                                     closingSymbol="RP", closingRegex="[)]" )
                P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
                
                
        The input "f(2)" would generate this tree::
        
                FUNCCALL
                 /    \\     
              ID(f)   NUM(2)
        
        
        To allow one or more arguments::
        
            P = syparser.Parser()
            P.addOperand(sym="ID", regex="[a-z]+")
            P.addOperator(sym="COMMA", regex=",", precedence=0)
            P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                                 closingSymbol="RP", closingRegex="[)]" )
            P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
     
        The input "f(2,3,4)" would generate this tree::
        
                FUNCCALL
                 /    \\     
              ID(f)   COMMA
                      /   \\
                  COMMA  NUM(4)
                  /   \\
               NUM(2) NUM(3)   
     
       
        To get zero-argument functions, we can use another implicit operator::
            
                P = syparser.Parser()
                P.addOperand(sym="ID", regex="[a-z]+")
                P.addOperator(sym="ADD", regex="[+]", precedence=10)
                P.addOperator(sym="COMMA", regex=",", precedence=-1000 )
                
                #this prevents the NOARGS from being synthesized in an empty
                #set of parens that appears somewhere other than a function call
                #context. For example,  
                #   "f()"
                #produces the token sequence "ID", "FUNCCALL", "LP", "NOARGS", "RP"
                #which then gives this parse tree:
                #           FUNCCALL
                #           /      \\
                #         ID (f)   NOARGS
                #But the input
                #   "x + ()"
                #produces the token sequence "ID", "ADD"
                #(and would then generate a parse error).
                P.addImplicitOperand(sym="NOARGS", symbolBefore=["FUNCCALL","LP"], 
                    symbolAfter="RP")
                
                P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                                     closingSymbol="RP", closingRegex="[)]" )
                                     
                #an ID immediately followed by a LP is interpreted as a function call
                P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
         
         
        The input "f(2,4)" would generate this tree::
        
                FUNCCALL
                 /    \\     
              ID(f)   COMMA
                      /   \\  
                   NUM(2) NUM(4)

        Or, to allow multiplication by putting two variables next to each other as in algebraic
        notation::
         
                P = syparser.Parser()
                P.addOperand(sym="ID", regex="[a-z]+")
                P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                                     closingSymbol="RP", closingRegex="[)]" )
                P.addImplicitOperator( sym="MUL", symbolBefore="ID", symbolAfter="ID")
                P.addImplicitOperator( sym="MUL", symbolBefore="RP", symbolAfter="LP")
         
        This would allow inputs like::
        
            x y
            (x)(y)
        
            
        Args:
        
            sym (str) : The symbol to be used in the parse tree for the implicit operator
            precedence (int): The precedence of the operator. In the case of functions,
                            this should probably be the highest precedence.
            symbolBefore (str|list[str]): If this is a string, it is interpreted as
                            the symbol which appears right before the implicit operator.
                            If this is a list of strings, it gives the sequence of
                            symbols that must appear before the implicit operator.
                            For example, ["ID","LP"] would require that 
                            the implicit operator can only be inserted
                            after the two tokens "ID" and "LP".
            symbolAfter (str):  The symbol which appears right after the implicit operator
            leftAssociative (bool): True if the implicit operator is left associative
            
            """
        t = _ImplicitOperatorTerminal(sym=sym, 
                precedence=precedence,
                symbolBefore=symbolBefore,
                symbolAfter=symbolAfter
        )
        self.implicitEntities.append(t)
        
    def addImplicitOperand(self,*,sym: str,symbolBefore: str,symbolAfter:str) -> None:
        """Add an implicit operand. This is useful for things like function calls that
        need to support empty parameter lists. The implicit operand will be created 
        automatically. For an example, see the code under :meth:`addImplicitOperator`.
        
        """
        t = _ImplicitOperandTerminal(sym=sym, 
                symbolBefore=symbolBefore,
                symbolAfter=symbolAfter
        )
        self.implicitEntities.append(t)
    
    def addComment(self,regex:Union[str,Pattern]) -> None:
        r""" Add a pattern that matches a comment. Comments are discarded
            and do not appear in the token stream.
            Example: Discard Python and C++ style comments from the input stream::
            
                P = syparser.Parser()
                P.addComment( r"#[^\n]*")
                P.addComment( r"/\*.*?\*/")
            
           
            Args:
                regex (str|regex): The regular expression for the symbol.
                               If a string is provided, it is interpreted 
                               as a regular expression. 
        """
        regex = _makeRegex(regex)
        self.terminals.append(  _CommentTerminal(sym= typing.cast(str,None), regex=regex ))
    
    def setWhitespace(self,regex:Union[str,Pattern]) -> None :
        r""" Set the pattern for whitespace. This defaults to "\\s+".
            Whitespace does not appear in the token stream.
           
            Args:
                regex (str|regex): The regular expression for the symbol.
                               If a string is provided, it is interpreted 
                               as a regular expression. 
        """
        regex = _makeRegex(regex) 
        for i in range(len(self.terminals)):
            if isinstance(self.terminals[i],_WhitespaceTerminal):
                self.terminals[i] = _WhitespaceTerminal(sym="WHITESPACE", regex=regex)
                return
       
    @staticmethod
    def createStandardParser():
        """Create a standard parser with most of the common
            operators predefined. The operators are these, in ascending
            order of priority:

            * Comma (,)
            * Assignment (=)
            * Logical or (||)
            * Logical and (&&)
            * Bitwise or (|)
            * Bitwise xor (^)
            * Bitwise and (&)
            * Equals and not-equals (==,!=)
            * Relational (>,>=,<,<=)
            * Shift (<<,>>)
            * Addition (+,-)
            * Multiplication, division, modulo (\\*,/,%)
            * Exponentiation (`**`)
            * Unary plus and minus (+,-)
            * Logical not, bitwise not (!,~)
            * Function call ( implicit; symbol=FUNCCALL )
            
            A function call with no arguments will have a synthetic "NOARGS" symbol as the
            second child of the FUNCCALL node (the first child is always the function name).
                
            Brackets are used for array indexing; an ARRAY operator is synthesized.
            
            Operands are floating point numbers or identifiers.
        
            Comments are denoted with #
            
        Returns:
            :class:`Parser`: A Parser object
        """
            
        P = Parser()
        P.setWhitespace("\\s+")
        P.addComment(regex="#[^\n]*")
        P.addOperator(sym="COMMA",regex=",",precedence=0)
        P.addOperator(sym="ASSIGN",regex="=",precedence=10)
        P.addOperator(sym="OR",regex="\|\|",precedence=20)
        P.addOperator(sym="AND",regex="&&",precedence=30)
        P.addOperator(sym="BITWISEOR",regex="\|",precedence=40)
        P.addOperator(sym="BITWISEXOR",regex="\^",precedence=50)
        P.addOperator(sym="BITWISEAND",regex="&",precedence=60)
        P.addOperator(sym="EQUALS",regex="==|!=",precedence=70)
        P.addOperator(sym="RELATIONAL",regex=">=|<=|>|<",precedence=80)
        P.addOperator(sym="SHIFT",regex="<<|>>",precedence=90)
        P.addOperator(sym="ADDOP",regex="[-+]",precedence=100)
        P.addOperator(sym="MULOP",regex="[*/%]",precedence=110)
        P.addOperator(sym="POWOP",regex="[*]{2}",precedence=120,leftAssociative=False)
        
        P.addNegationStyleOperator(sym="PLUSMINUS",transformFromSym="ADDOP",precedence=130)
        P.addOperator(sym="NOT", regex="!", precedence=130,arity=1,leftAssociative=False)
        P.addOperator(sym="BITNOT", regex="~", precedence=130,arity=1,leftAssociative=False)
        
        P.addGroupingSymbol(openingSymbol="LB",openingRegex="\[",
            closingSymbol="RB",closingRegex="\]")
            
        #empty parens will have a NOARGS token synthesized
        P.addGroupingSymbol(openingSymbol="LP",openingRegex="[(]",
            closingSymbol="RP",closingRegex="[)]")
        P.addImplicitOperator(sym="FUNC",symbolBefore="ID",symbolAfter="LP",precedence=500)
        P.addImplicitOperator(sym="NOARGS",symbolBefore=["FUNC","LP"],symbolAfter="RP")

        #empty brackets are not allowed 
        P.addImplicitOperator(sym="ARRAY",symbolBefore="ID",symbolAfter="LB")
        
        P.addOperand( sym="ID", regex="[A-Za-z_]\w*")
        P.addOperand( sym="NUM", regex=r"(\d+|\d+\.\d*|\.\d+)([Ee][-+]?\d+)?")
    
        
    @staticmethod
    def toDot(root: TreeNode,fp: TextIO) -> None:
        """Output a parse tree in Graphviz 'dot' format.
        
        Args:
            root (:class:`TreeNode`): Root of the parse tree
            fp: A file stream. The data will be written here.
        """
        
        def toDot1(n):
            fp.write(f'{n.unique} [label="{n.sym}')
            if n.token and n.token.lexeme:
                fp.write("\\n")
                fp.write(n.token.lexeme.replace('"','\\"'))
            fp.write('"];\n')
            for c in n.children:
                toDot1(c)
        def toDot2(n):
            for c in n.children:
                fp.write(f"{n.unique} -- {c.unique};\n")
            for c in n.children:
                toDot2(c)
        fp.write("graph d {\n")
        toDot1(root) 
        toDot2(root) 
        fp.write("}\n")

    @staticmethod
    def toText(root:TreeNode) -> str:
        """Produce a text representation of a given parse tree.
        Useful for debugging. Example: The parse tree for "(1+2)*3" might be::
        
            MULOP (*)
            |
            +----ADDOP (+)
            |    |
            |    +----NUM (1)
            |    |
            |    +----NUM (2)
            |
            +----NUM (3)
        
        Or, the parse tree for "f(x,y,z+w)" might be::
        
            FUNCCALL (None)
            |
            +----ID (f)
            |
            +----COMMA (,)
                 |
                 +----COMMA (,)
                 |    |
                 |    +----ID (x)
                 |    |
                 |    +----ID (y)
                 |
                 +----ADDOP (+)
                      |
                      +----ID (z)
                      |
                      +----ID (w)

        
        Args:
            root (TreeNode): The root of the parse tree
        Returns:
            str: The output string.
        """
    
        L=[""]
        
        def helper(n:TreeNode, nesting):
            pfx = ""
            for thing in nesting:
                if thing=="y":
                    pfx += "|    "
                else:
                    pfx += "     "
            for i,c in enumerate(n.children):
                L.append(pfx+"|")
                L.append("{}+----{} ({})".format(pfx,c.sym,c.lexeme))
                helper(c,nesting + ("y" if i < len(n.children)-1 else "n") )
            
        L.append("{} ({})".format(root.sym,root.lexeme))
        helper(root,"")
        return "\n".join(L)
   
        
        

def main():
            
    import sys
    import inspect
    
    
    if len(sys.argv) > 1 and sys.argv[1] == "--doc":
        #generate doc index.rst
        with open("docs/source/index.rst","w") as fp:
            fp.write(".. This file is auto-generated by syparser.py\n\n")
            fp.write(""".. syparser documentation master file, created by
sphinx-quickstart on Wed Jul 28 11:11:48 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

syparser: Shunting Yard Parser
====================================
 
    
.. toctree::
   :caption: Contents
   :maxdepth: 3
   
""")
         
            G=globals()
            k=list(globals().keys())
            lst=[]
            for k in G:
                if k.startswith("_"):
                    continue
                if k in ["List", "Union", "Pattern", "Optional", "TextIO", "Generic", "TypeVar"]:
                    continue
                if inspect.isclass( G[k] ):
                    lst.append(k)
                    
            lst.sort()
            
            for name in lst:
                fp.write("\n")
                fp.write("{}\n".format(name))
                fp.write("="*len(name))
                fp.write("\n\n")
                fp.write(f".. autoclass:: syparser.{name}\n")
                fp.write("    :members:\n")
                fp.write("    :inherited-members:\n")
                fp.write("    :show-inheritance:\n")
                fp.write("    :exclude-members: with_traceback\n")
                fp.write("\n")
                
            fp.write("""
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


""")    
    
            
        
    logging.basicConfig(level=logging.DEBUG)
    class X:
        def __init__(self,lbl):
            self.sym=lbl
            self.children=[]
            self.lexeme="fakeLexeme"
        def addChild(self,c):
            self.addChild(c)
       
                 
    """
            foo                 pfx=""
            |                   pfx=""
            +----left           pfx=""
            |    |              pfx=
            |    +----a
            |    |    |
            |    |    +----c
            |    |
            |    +----b
            |
            +----right
            
          """  
    if 0:          
        foo = X("foo")
        left = X("left")
        right = X("right")
        a = X("a")
        c = X("c")
        b = X("b")
        foo.addChild(left)
        foo.addChild(right)
        left.addChild(a)
        a.addChild(c)
        left.addChild(b)
        print(Parser.toText(foo))
    
    if 0:          
        A = X("A")
        B = X("B")
        C = X("C")
        D = X("D")
        A.addChild(B)
        B.addChild(C)
        B.addChild(D)
        print(Parser.toText(A)) 
        
    if 0:
        P = Parser()
        P.addOperand(sym="ID", regex="[a-z]+")
        P.addOperator(sym="ADDOP", regex="[+]", precedence=10)
        P.addGroupingSymbol( openingSymbol="LP", openingRegex="[(]", 
                             closingSymbol="RP", closingRegex="[)]" )
        P.addImplicitOperand(sym="VOID",symbolBefore="LP", symbolAfter="RP")
        P.addOperator(sym="COMMA", regex=",", precedence=0)
        P.addImplicitOperator( sym="FUNCCALL", symbolBefore="ID", symbolAfter="LP")
        root = P.parse("f()")
        print(Parser.toText(root))
        
    if 0:
        P = Parser()
        P.addOperator("PLUS","[!@#$%^&*+]",precedence=10)
        P.addOperator("MUL","[*]",precedence=20)
        P.addOperand(sym="NUM", regex="-?\\d+")
        P.addGroupingSymbol( openingSymbol="ABS", openingRegex=r"\|", 
                             closingSymbol="ABS", closingRegex=r"\|",
                             addToTree=True )
        P.addGroupingSymbol( openingSymbol="LP", openingRegex=r"\(", 
                             closingSymbol="RP", closingRegex=r"\)" )
                             
        inp = "1 ! | |2@3| # |4$5| |"
        print(Parser.toText(P.parse(inp)))
        print(inp)
        
    if 0:
        P = Parser()
        P.setWhitespace("\\s+")
        P.addOperator(sym="ADDOP",regex="[-+]",precedence=1)
        P.addOperator(sym="MULOP",regex="[*/]",precedence=2)
        P.addOperator(sym="POWOP",regex="[*]{2}",precedence=10)
        P.addOperator(sym="COMMA",regex=",",precedence=0)
        P.addNegationStyleOperator(sym="NEGATE",transformFromSym="ADDOP",precedence=5)
        P.addGroupingSymbol(openingSymbol="LP",openingRegex="[(]",closingSymbol="RP",closingRegex="[)]")
        P.addGroupingSymbol(openingSymbol="LB",openingRegex="\\[",closingSymbol="RB",
                            closingRegex="\\]")
    
        P.addImplicitOperator(sym="FUNCCALL",symbolBefore="ID",symbolAfter="LP")
        P.addImplicitOperator(sym="ARRAY",symbolBefore="ID",symbolAfter="LB")
        P.addImplicitOperator(sym="ARRAY2",symbolBefore="RB",symbolAfter="LB")
        # ~ P.addImplicitOperator(sym="IMUL",symbolBefore="ID",symbolAfter="ID")
        P.addOperand( sym="ID", regex="[A-Za-z]+")
        P.addOperand( sym="NUM", regex="\\d+" )
        P.addComment(regex="#[^\n]*")
    
        if len(sys.argv) > 1:
            inp = sys.argv[1]
        else:
            inp = "1+-x*foo(1,2,baz[42][7])"
        
        print("Input:",inp)
        root = P.parse( inp )
        with open("test.dot","w") as fp:
            Parser.toDot(root,fp)
        print(Parser.toText(root))
        print("Write test.dot for input -->"+inp)

# implicit multiply:
#  x y
# assignment
#   z = x y
#   z = x * y 

if __name__ == "__main__":
    main()
