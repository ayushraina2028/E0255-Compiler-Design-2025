#include <bits/stdc++.h>

enum Token {
    Token_EndOfFile = -1, 

    // Commands
    Token_Def = -2,
    Token_Extern = -3, 

    // Primary
    Token_Identifier = -4,  // Fixed Typo
    Token_Number = -5,      // Consistent Naming
};

static std::string IdentifierString; // Gets filled if Current Token is an Identifier
static double numValue; // Gets Filled if Current Token is a number

// Get_Token function generates the next Token from standard input
static int Get_Token() {
    static int LastChar = ' ';

    // Skip whitespaces
    while (isspace(LastChar))
        LastChar = getchar();  

    // Identifier or keyword
    if (isalpha(LastChar)) {
        IdentifierString = LastChar;

        while (isalnum(LastChar = getchar())) {
            IdentifierString += LastChar;
        }

        if (IdentifierString == "def") return Token_Def;
        if (IdentifierString == "extern") return Token_Extern;
        
        return Token_Identifier;
    }

    // Number parsing
    if (isdigit(LastChar) || LastChar == '.') {
        std::string NumberString;
        
        do {
            NumberString += LastChar;
            LastChar = getchar();
        } while (isdigit(LastChar) || LastChar == '.');

        numValue = strtod(NumberString.c_str(), nullptr);
        return Token_Number;
    }

    // Handling comments
    if (LastChar == '#') {
        do {
            LastChar = getchar();
        } while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

        if (LastChar != EOF) return Get_Token();
    }

    // End of file
    if (LastChar == EOF)
        return Token_EndOfFile;

    // Return single character tokens
    int ThisChar = LastChar;
    LastChar = getchar();
    return ThisChar;
}


namespace {
    
    /* Base Class for All Expression Nodes */
    class ExpressionAST {
    public:
        virtual ~ExpressionAST() = default;
    };

    /* Expression Class for Numeric Literals */
    class NumberExpressionAST : public ExpressionAST {
        // Public Members of ExpressionAST remain Public in NumberExpressionAST
    private:
        double Value;

    public:
        // Constructor
        NumberExpressionAST(double Value) {
            this->Value = Value;
        }
    };

    /* Expression Class for referencing a variable */
    class VariableExpressionAST : public ExpressionAST {
    private:
        std::string Name;
    
    public:
        VariableExpressionAST(const std::string& String) {
            this->Name = String;
        }
    };

    /* Expression class for binary operator */
    class BinaryExpressionAST : public ExpressionAST {
    private:
        char Op; // Operation
        std::unique_ptr<ExpressionAST> LHS, RHS; // Operands
 
    public:
        BinaryExpressionAST(char Op, std::unique_ptr<ExpressionAST> LHS, std::unique_ptr<ExpressionAST> RHS) {
            this->Op = Op;
            this->LHS = std::move(LHS);
            this->RHS = std::move(RHS);
        }
    };

    /* Expression Class for function calls */
    class CallExpressionAST : public ExpressionAST {
    private:
        std::string Callee;
        std::vector<std::unique_ptr<ExpressionAST>> Arguments;

    public:
        CallExpressionAST(const std::string& Callee, std::vector<std::unique_ptr<ExpressionAST>> Arguments) {
            this->Callee = Callee;
            this->Arguments = std::move(Arguments);
        }
    };

    /* This Class represents the prototype for a function */
    class PrototypeAST {
    private:
        std::string Name;
        std::vector<std::string> Arguments;

    public: 
        PrototypeAST(const std::string& Name, std::vector<std::string> Arguments) { /* Function name is passed by Reference */
            this->Name = Name;
            this->Arguments = std::move(Arguments);
        }

        const std::string& getName() const {
            return Name;
        }
    };

    /* This class represents a function definition */
    class FunctionAST {
    private:
        std::unique_ptr<PrototypeAST> Proto;
        std::unique_ptr<ExpressionAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto, std::unique_ptr<ExpressionAST> Body) {
            this->Proto = std::move(Proto);
            this->Body = std::move(Body);
        }
    };

} /* namespace ends here */

/* Building Parser */
static int CurrentToken;

static int GetNextToken() {
    return CurrentToken = Get_Token();
}

/* Binop Precedence holds precedence of binary operators which are defined */
static std::map<char, int> BinopPrecedence;

/* This function returns the precedence of pending binary operator token */
static int GetTokenPrecedence() {
    if(!isascii(CurrentToken)) return -1;

    int TokenPrecedence = BinopPrecedence[CurrentToken];
    if(TokenPrecedence <= 0) return -1;
    
    return TokenPrecedence;
}

/* 2 Error Handling Function for ExprAST and PrototypeAST */
std::unique_ptr<ExpressionAST> LogError(const char* ch) {
    fprintf(stderr, "Error: %s\n", ch);
    return nullptr; /* null pointer wrapped in unique pointer */
} 

std::unique_ptr<PrototypeAST> LogErrorP(const char* ch) {
    LogError(ch);
    return nullptr;
}

static std::unique_ptr<ExpressionAST> ParseExpressions();

/* Function to parse a numeric literal(number). Gets Called when current token is Token_Number */
static std::unique_ptr<NumberExpressionAST> ParseNumberExpression() {
    
    auto Result = std::make_unique<NumberExpressionAST>(numValue);
    GetNextToken(); 

    return  std::move(Result);
}

/* Function to parse expressions inside parenthesis  '(' and ')' */
static std::unique_ptr<ExpressionAST> ParseParenthesizedExpression() {
    GetNextToken();
    auto V = ParseExpressions();

    if(!V) return nullptr;
    if(CurrentToken != ')') return LogError("expected ')'");

    GetNextToken();
    return V;
}

/* Function to parse expressions that start with an identifier */
static std::unique_ptr<ExpressionAST> ParseIdentifierExpression() {
    std::string IDname = IdentifierString;
    GetNextToken();

    if(CurrentToken != '(') {
        return std::make_unique<VariableExpressionAST>(IDname);
    }

    GetNextToken();
    std::vector<std::unique_ptr<ExpressionAST>> Arguments;

    if(CurrentToken != ')') {
        while(true) {
            
            if(auto argument = ParseExpressions()) Arguments.push_back(std::move(argument));
            else return nullptr;

            if(CurrentToken == ')') break;
            if(CurrentToken != ',') return LogError("Expected ')' or ',' in Argument List");

            GetNextToken();
        }
    }

    GetNextToken(); /* Eat ) */
    return std::make_unique<CallExpressionAST>(IDname, std::move(Arguments));
}

/* Function to parsing primary expressions - Variables, Function Calls, Numeric Literals, Parenthesized Expressions */
static std::unique_ptr<ExpressionAST> ParsePrimaryExpressions() {
    switch(CurrentToken) {

        default:
            return LogError("unknown token while parsing the expression");
        case Token_Identifier:
            return ParseIdentifierExpression();
        case Token_Number:
            return ParseNumberExpression();
        case '(':
            return ParseParenthesizedExpression();

    }
}

static std::unique_ptr<ExpressionAST> ParseBinaryOperationRHS(int ExpressionPrecedence, std::unique_ptr<ExpressionAST> LHS) {
    while(true) {

        int TokenPrecedence = GetTokenPrecedence();
        if(TokenPrecedence < ExpressionPrecedence) return LHS;

        // Now we are sure this is a binop
        int BinaryOperation = CurrentToken;
        GetNextToken(); // Eat

        auto RHS = ParsePrimaryExpressions();
        if(!RHS) return nullptr;

        int NextPrecedence = GetTokenPrecedence();
        if(TokenPrecedence < NextPrecedence) {
            RHS = ParseBinaryOperationRHS(TokenPrecedence+1, std::move(RHS)); // Recursion
            if(!RHS) return nullptr;
        }

        // Merge
        LHS = std::make_unique<BinaryExpressionAST>(BinaryOperation, std::move(LHS), std::move(RHS));

    }
}

static std::unique_ptr<ExpressionAST> ParseExpressions() {
    auto LHS = ParsePrimaryExpressions();
    if(!LHS) return nullptr;

    return ParseBinaryOperationRHS(0, std::move(LHS));   
}

/* Function to parse prototype */
static std::unique_ptr<PrototypeAST> ParsePrototype() {
    if(CurrentToken != Token_Identifier) return LogErrorP("Expected Function Name in prototype");
    
    std::string FunctionName = IdentifierString;
    GetNextToken();

    if(CurrentToken != '(') return LogErrorP("Expected '(' in prototype");

    /* Read List of Argument Names */
    std::vector<std::string> Arguments;
    while(GetNextToken() == Token_Identifier) Arguments.push_back(IdentifierString);

    if(CurrentToken != ')') return LogErrorP("Expected ')' in prototype");

    /* Successfully parsed, eat ) now */
    GetNextToken(); 
    return std::make_unique<PrototypeAST>(FunctionName, std::move(Arguments));
}

/* This Function is for parsing definition of function expression (def prototype) */
static std::unique_ptr<FunctionAST> ParseFunctionDefn() {
    GetNextToken();
    auto Proto = ParsePrototype();

    if(!Proto) return nullptr;

    if(auto E = ParseExpressions()) {
        return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
    }
    return nullptr;
}

/* Function to parse extern prototype */
static std::unique_ptr<PrototypeAST> ParseExtern() {
    GetNextToken(); /* eat extern */
    return ParsePrototype();
}

/* Function to parse top level expressions */
static std::unique_ptr<FunctionAST> ParseTopLevelExpression() {
    if(auto E = ParseExpressions()) {
        auto Proto = std::make_unique<PrototypeAST>("", std::vector<std::string>());
        return std::make_unique<FunctionAST> (std::move(Proto), std::move(E));
    }   
    return nullptr;
}

/* Functions used for Top Level Parsing */

static void HandleDefinitions() {
    if(ParseFunctionDefn()) {
        fprintf(stderr, "Parsed a Function Definition. \n");
    }
    else {
        GetNextToken();
    }
}

static void HandleExtern() {
  if (ParseExtern()) {
    fprintf(stderr, "Parsed an extern\n");
  } else {
    // Skip token for error recovery.
    GetNextToken();
  }
}

static void HandleTopLevelExpression() {
  // Evaluate a top-level expression into an anonymous function.
  if (ParseTopLevelExpression()) {
    fprintf(stderr, "Parsed a top-level expr\n");
  } else {
    // Skip token for error recovery.
    GetNextToken();
  }
}

/// top ::= definition | external | expression | ';'
static void MainLoop() {
  while (true) {
    fprintf(stderr, "ready> ");
    switch (CurrentToken) {
    case Token_EndOfFile:
      return;
    case ';': // ignore top-level semicolons.
      GetNextToken();
      break;
    case Token_Def:
      HandleDefinitions();
      break;
    case Token_Extern:
      HandleExtern();
      break;
    default:
      HandleTopLevelExpression();
      break;
    }
  }
}

int main() {
    BinopPrecedence['<'] = 10;
    BinopPrecedence['+'] = 20;
    BinopPrecedence['-'] = 20;
    BinopPrecedence['*'] = 40;

        // Prime the first token.
    fprintf(stderr, "ready> ");
    GetNextToken();

    // Run the main "interpreter loop" now.
    MainLoop();
}