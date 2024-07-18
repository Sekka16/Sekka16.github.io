---
title: Tree-sitter-01 环境配置 
date: 2024-7-18 16:20:00
tags:
categories:
description: 
---


# Tree-sitter安装与使用

## Tree-sitter环境配置

```bash
# This will prompt you for input
npm init
# This installs a small module that lets your parser be used from Node
npm install --save nan
# This installs the Tree-sitter CLI itself
npm install --save-dev tree-sitter-cli 
# 为啥使用 --save-dev 可参考
# https://stackoverflow.com/questions/22891211/what-is-the-difference-between-save-and-save-dev

# 克隆C语言文法
git clone https://github.com/tree-sitter/tree-sitter-c.git && cd tree-sitter-c

# 生成parser.c ? 暂时不知道什么作用
tree-sitter generate
```

## 解析`hello.c` 

```c
#include <stdio.h>

int main() {
    prinf("hello world\n");
    return 0;
}
```

```bash
tree-sitter parse hello.c
```

输出如下：
```bash
sekka•tree-sitter/tree-sitter-hello/tree-sitter-c(master⚡)» tree-sitter parse hello.c                                                                                    [16:23:53]
(translation_unit [0, 0] - [6, 0]
  (preproc_include [0, 0] - [1, 0]
    path: (system_lib_string [0, 9] - [0, 18]))
  (function_definition [2, 0] - [5, 1]
    type: (primitive_type [2, 0] - [2, 3])
    declarator: (function_declarator [2, 4] - [2, 10]
      declarator: (identifier [2, 4] - [2, 8])
      parameters: (parameter_list [2, 8] - [2, 10]))
    body: (compound_statement [2, 11] - [5, 1]
      (expression_statement [3, 2] - [3, 27]
        (call_expression [3, 2] - [3, 26]
          function: (identifier [3, 2] - [3, 8])
          arguments: (argument_list [3, 8] - [3, 26]
            (string_literal [3, 9] - [3, 25]
              (string_content [3, 10] - [3, 22])
              (escape_sequence [3, 22] - [3, 24])))))
      (return_statement [4, 2] - [4, 11]
        (number_literal [4, 9] - [4, 10])))))
```

## 生成LLVM IR

```python
import subprocess
import json
from llvmlite import ir

# Step 1: Parse the C code using Tree-sitter
def parse_c_code(file_path):
    result = subprocess.run(['tree-sitter', 'parse', file_path], capture_output=True, text=True)

    # Debug: print the raw output and error
    print("Tree-sitter output:", result.stdout)
    # print("Tree-sitter error:", result.stderr)
    print("Return code:", result.returncode)

    # Here, instead of parsing JSON, we directly use the raw output
    if result.returncode != 0:
        print("Tree-sitter failed to parse the file.")
        return None

    # Assuming the output is in a format we can work with directly
    return result.stdout

# Step 2: Generate LLVM IR from the parsed AST
def generate_llvm_ir(ast):
    if ast is None:
        print("AST is None, cannot generate LLVM IR.")
        return ""

    module = ir.Module(name='hello_module')

    # Define the function
    func_type = ir.FunctionType(ir.IntType(32), [])
    func = ir.Function(module, func_type, name='main')
    block = func.append_basic_block(name='entry')
    builder = ir.IRBuilder(block)

    # Create global string for "Hello, World!\n"
    hello_str = ir.GlobalVariable(module, ir.ArrayType(ir.IntType(8), 15), name="hello_str")
    hello_str.initializer = ir.Constant(ir.ArrayType(ir.IntType(8), 15), bytearray("Hello, World!\n\0", 'utf8'))
    hello_str.global_constant = True
    hello_str.linkage = 'internal'

    # Get a pointer to the first element of the string
    str_ptr = builder.gep(hello_str, [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0)], inbounds=True)

    # Declare printf function
    printf_ty = ir.FunctionType(ir.IntType(32), [ir.PointerType(ir.IntType(8))], var_arg=True)
    printf = ir.Function(module, printf_ty, name='printf')

    # Call printf function
    builder.call(printf, [str_ptr])

    # Return 0
    builder.ret(ir.Constant(ir.IntType(32), 0))

    return str(module)

# Main function to run the steps
if __name__ == '__main__':
    # Parse the C file to get the AST
    ast = parse_c_code('hello.c')
    print("Parsed AST:")
    if ast is not None:
        print(ast)
    else:
        print("Failed to parse AST.")

    # Generate LLVM IR from the AST
    llvm_ir = generate_llvm_ir(ast)
    print("\nGenerated LLVM IR:")
    print(llvm_ir)
```

输出结果如下：

```bash
sekka•tree-sitter/tree-sitter-hello/tree-sitter-c(master⚡)» python3 hello.py                                                                                             [16:30:00]
Parsed AST:
(translation_unit [0, 0] - [6, 0]
  (preproc_include [0, 0] - [1, 0]
    path: (system_lib_string [0, 9] - [0, 18]))
  (function_definition [2, 0] - [5, 1]
    type: (primitive_type [2, 0] - [2, 3])
    declarator: (function_declarator [2, 4] - [2, 10]
      declarator: (identifier [2, 4] - [2, 8])
      parameters: (parameter_list [2, 8] - [2, 10]))
    body: (compound_statement [2, 11] - [5, 1]
      (expression_statement [3, 2] - [3, 27]
        (call_expression [3, 2] - [3, 26]
          function: (identifier [3, 2] - [3, 8])
          arguments: (argument_list [3, 8] - [3, 26]
            (string_literal [3, 9] - [3, 25]
              (string_content [3, 10] - [3, 22])
              (escape_sequence [3, 22] - [3, 24])))))
      (return_statement [4, 2] - [4, 11]
        (number_literal [4, 9] - [4, 10])))))


Generated LLVM IR:
; ModuleID = "hello_module"
target triple = "unknown-unknown-unknown"
target datalayout = ""

define i32 @"main"()
{
entry:
  %".2" = getelementptr inbounds [15 x i8], [15 x i8]* @"hello_str", i32 0, i32 0
  %".3" = call i32 (i8*, ...) @"printf"(i8* %".2")
  ret i32 0
}

@"hello_str" = internal constant [15 x i8] c"Hello, World!\0a\00"
declare i32 @"printf"(i8* %".1", ...)
```
