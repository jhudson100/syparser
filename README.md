syparser
==========

This module contains a simple parser for mathematical expressions. 
It's pretty flexible and can be used for a lot more than that.


Basic example
--------------

    P = syparser.Parser()
    P.addOperator(sym="ADD", precedence=10, regex="[-+]")
    P.addOperator(sym="MUL", precedence=20, regex="[*/]")
    P.addOperator(sym="POW", precedence=30, regex="[*]{2}", leftAssociative=False)
    P.addGroupingSymbol(openingSymbol="LP", openingRegex="[(]",
        closingSymbol="RP", closingRegex="[)]")
    P.addOperand(sym="NUM", regex="[0-9]+")


Now that the parser has been created, it can be used to parse strings:

    root = P.parse("(1+2)*3**4")


This returns the root of a parse tree. For this example,
the tree looks like this::

```
                   MUL(*)
                 /       \
           ADD(+)        POW(**)
          /     \        /     \
        NUM(1) NUM(2)  NUM(3) NUM(4)
```        
