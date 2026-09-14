## Example Grammar Structure Statements & Expressions

This is a statement, which are high-level instructions which are semantically clear to the user

capacity <- length ptr 0
capacity <- length ptr myVar
capacity <- length ptr fooBar()

This can be simplified to:

VAR <- EXP

And furthermore:

VAR <- length VAR EXPR

## Another Example

m <- new Array(4, 4)

VAR <- EXPR

number <- number < 1

VAR <- EXPR

That's also:

VAR <- EXPR COP EXPR

# Summary

Understanding that expressions are like:

1 + 2

3 + x

fooBar()

Understanding that expressions statements are like

fooBar()

myVar++

Ideally the grammar is extremely modular and simple. The PEG is too hardcoded right now, for instance:

length_expr = variable WS "<-" WS "length" WS variable (WS number)? WS_AND_COMMENTS

Can be greatly improved by defining the expression first:

lenExpr = "length" WS variable (WS number)?

Also note that "... variable (WS number)?" can also be improved, we're not simply limited to numbers

