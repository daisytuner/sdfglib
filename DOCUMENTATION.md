# sdfglib Documentation

The sdfglib codebase uses [Doxygen](https://www.doxygen.nl/) for documentation generation. Doxygen is the de-facto standard for generating documentation from C++ source code.

## Building Documentation

### Prerequisites

Install Doxygen and Graphviz (for diagrams):

```bash
sudo apt-get install -y doxygen graphviz
```

### Generate Documentation

From the repository root:

```bash
doxygen Doxyfile
```

This will generate documentation in the `docs/` directory.

### View Documentation

Open the generated HTML documentation in a web browser:

```bash
# Linux/WSL
xdg-open docs/html/index.html

# macOS
open docs/html/index.html

# Or simply navigate to file:///<repo-path>/docs/html/index.html in your browser
```

## Documentation Structure

The generated documentation includes:

- **Class List**: All documented classes with brief descriptions
- **File List**: All documented source files
- **Namespace List**: Organized by namespace (sdfg::types, sdfg::symbolic)
- **Class Hierarchy**: Inheritance relationships
- **Class Members**: Detailed member documentation with parameters and return values
- **Module Groups**: Functional groupings (e.g., symbolic arithmetic, symbolic logic)

## Documented Components

### Type System (include/sdfg/types)

The type system is fully documented with Doxygen comments, covering:
- Primitive types (integers, floats, void, bool)
- Composite types (arrays, pointers, structures, functions)
- Storage types and memory management
- Type operations and utilities

### Symbolic System (include/sdfg/symbolic)

The symbolic expression system is comprehensively documented, covering:
- **symbolic.h**: Core symbolic expressions, symbols, and operations
- **assumptions.h**: Symbol assumptions for bounds, constness, and evolution
- **extreme_values.h**: Computing bounds of expressions
- **sets.h**: Integer set operations and disjointness checking
- **maps.h**: Symbol evolution and monotonicity analysis
- **polynomials.h**: Polynomial representation and coefficient extraction
- **utils.h**: ISL (Integer Set Library) integration
- **series.h**: Analysis of symbolic series and sequences
- **conjunctive_normal_form.h**: CNF conversion for boolean conditions

## Configuration

The documentation is configured via `Doxyfile` in the repository root. Key settings:

- `INPUT = include/sdfg/types include/sdfg/symbolic` - Documents type and symbolic systems
- `OUTPUT_DIRECTORY = docs` - Output location
- `RECURSIVE = YES` - Process subdirectories
- `EXTRACT_ALL = YES` - Document all entities (even without explicit doc comments)

To expand documentation to other parts of sdfglib, modify the `INPUT` setting in `Doxyfile`.

## Website Generation

The documentation can be published as a static website by:

1. Hosting the `docs/html/` directory on a web server
2. Using GitHub Pages (copy/deploy the html directory to gh-pages branch)
3. Using documentation hosting services like Read the Docs or GitBook

## Contributing Documentation

When adding new code to documented areas (type system, symbolic system, etc.):

1. Add Doxygen comments using `/** ... */` or `///` style
2. Document all public classes, methods, and functions
3. Include `@param` tags for parameters and `@return` tags for return values
4. Add `@brief` descriptions for classes and complex methods
5. Use `@code` blocks for examples
6. Rebuild documentation to verify formatting

### Example

```cpp
/**
 * @brief Converts a type to its string representation
 * @param type The type to convert
 * @return A human-readable string describing the type
 */
std::string type_to_string(const IType& type);
```

For more examples, see the existing documentation in:
- `include/sdfg/types/` for type system documentation style
- `include/sdfg/symbolic/` for symbolic system documentation style
