# Gemini & AI Assistant Guide for 322Tooling

This document provides high-density technical context for AI assistants
contributing to L-series interpreters for CS322.

## About

This project is a collection of interpreters for a series of custom low-level programming languages (L1, L2, L3, IR, LA, LB). The interpreters are written in Python and use the PEG (Parsing Expression Grammar) parsing technique to parse the L language syntax.

This is designed to be a tool for students in the CS322 course. Read the course description for more information:

```
The compiler is the programmer's primary tool. Understanding the compiler is therefore critical for programmers, even if they never build one. Furthermore, many design techniques that emerged in the context of compilers are useful for a range of other application areas. This course introduces students to the essential elements of building a compiler: parsing, context-sensitive property checking, code linearization, register allocation, etc. To take this course, students are expected to already understand how programming languages behave, to a fairly detailed degree. The material in the course builds on that knowledge via a series of semantics preserving transformations that start with a fairly high-level programming language and culminate in machine code.

    This course satisfies the Systems breadth and the project requirement.
```

## Design Philosophy

LB is the highest level language and will be implemented first. There should be shared components across the interpreters to minimize code duplication. The design philosophy is to keep the interpreters simple and modular; the interpreters are not indended to be production quality, but rather to be educational tools for students to learn about compiler design and implementation.

## Interpreter Design

This project uses Parsimonious. Refer to parsimonious_docs.md for more information.

The design ideally mirrors the grammar of the provided language documentation.

## General instructions

Do not perform any actions that are not requested. If you are unsure, ask for clarification.

Do not provide long explanations or summaries. Only provide relevant information.

Do not provide summaries of code or documentation.

Do not scan the directory structure or provide boilerplate code unless explicitly requested.

Do not comment in code unless explicitly requested.

Follow the the project design choices of the user.

Follow readable code style, formatting, and industry practices. Refer to the user for direction if you are unsure.

## Communications

-   **Communication**: Be concise, professional, and technical. Use GitHub-style
    markdown.


-   **Verification**: Always run relevant tests.


- **Git**: Do not push directly. Do not co-author commits.
