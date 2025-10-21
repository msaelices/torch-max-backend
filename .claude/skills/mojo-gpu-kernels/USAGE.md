# How to Use the Mojo GPU Kernels Skill

## Quick Start

The skill is automatically available in Claude Code sessions within this project. Simply reference it in your prompts.

## Example Prompts

### For Learning

```
Use the mojo-gpu-kernels skill to explain how warp reductions work.
```

```
What's the best pattern for implementing element-wise operations? Use the mojo-gpu-kernels skill.
```

### For Implementation

```
I need to implement the log_softmax operation for PyTorch. Use the mojo-gpu-kernels skill to help me:
1. Find if MAX has a similar operation
2. Choose the right kernel pattern
3. Write the implementation with proper type hints
```

```
Help me optimize this reduction kernel using the mojo-gpu-kernels skill.
```

```
Use the mojo-gpu-kernels skill to show me how to implement a 2D stencil operation with shared memory.
```

### For Debugging

```
I'm getting bank conflicts in my shared memory kernel. Use the mojo-gpu-kernels skill to show me how to fix this.
```

```
My kernel is slow. Can the mojo-gpu-kernels skill help me identify performance issues?
```

### For Following Project Workflow

```
I want to add support for the aten::relu operation. Use the mojo-gpu-kernels skill to guide me through the complete workflow from research to testing.
```

## When to Use the Skill

### Always Use For:
- Implementing new ATen operations
- Writing custom GPU kernels
- Optimizing existing kernels
- Understanding MAX framework patterns
- Following best practices

### The Skill Helps With:
1. **Pattern Selection**: Choosing the right kernel pattern for your operation
2. **Implementation**: Providing templates and examples
3. **Optimization**: Performance tips and best practices
4. **Integration**: MAX framework operation discovery
5. **Testing**: Test strategies and parametrization
6. **Debugging**: Common pitfalls and solutions

## Skill Structure

When you invoke the skill, it provides:

### Level 1: Overview
- Skill name and description
- When it's appropriate to use

### Level 2: Core Guidance (SKILL.md)
- Essential imports
- Kernel patterns with examples
- Type handling
- Performance considerations
- Best practices

### Level 3: Deep Dive (references/)
- Detailed pattern documentation
- Advanced techniques
- Complete examples
- Performance analysis

### Level 4: Templates (assets/)
- Copy-paste ready code
- Fully commented templates
- Launch configuration helpers
- Example usage

## Example Workflow

### Implementing a New ATen Operation

```
Prompt: Use the mojo-gpu-kernels skill to help me implement aten::gelu

Response includes:
1. Search for GELU in MAX operations (found in activations.mojo)
2. Recommend using built-in MAX gelu function
3. Show integration pattern with PyTorch backend
4. Provide test template
5. Show type hints
6. Explain numerical stability considerations
```

### Writing a Custom Kernel

```
Prompt: I need to write a custom kernel for parallel prefix sum. Use the mojo-gpu-kernels skill.

Response includes:
1. Recommend shared memory pattern
2. Provide reference to shared_memory_patterns.md
3. Show up-sweep and down-sweep algorithm
4. Include complete code example
5. Explain synchronization requirements
6. Provide launch configuration
7. Testing strategy
```

### Optimizing Performance

```
Prompt: My reduction kernel is slow. Use the mojo-gpu-kernels skill to help optimize it.

Response analyzes:
1. Current pattern vs. optimal pattern
2. Memory access patterns (coalescing)
3. Shared memory usage
4. Warp primitive opportunities
5. Occupancy considerations
6. Suggests specific improvements with code
```

## Skill Contents Reference

### Core Documentation
- `SKILL.md` - Main skill file with comprehensive guidance
- `README.md` - Skill overview and quick start
- `USAGE.md` - This file

### Pattern References
- `references/elementwise_patterns.md` - Element-wise operations
- `references/reduction_patterns.md` - Reduction patterns
- `references/shared_memory_patterns.md` - Shared memory usage
- `references/max_operations.md` - MAX framework operations catalog

### Code Templates
- `assets/elementwise_template.mojo` - Element-wise kernel templates
- `assets/reduction_template.mojo` - Reduction kernel templates
- `assets/activation_templates.mojo` - Neural network activations

## Tips

1. **Be Specific**: The more specific your prompt, the more targeted the help
   - Good: "Use mojo-gpu-kernels skill to implement log_softmax with numerical stability"
   - Less specific: "Help with softmax"

2. **Reference the Workflow**: Mention which step of the workflow you're on
   - "I'm at step 5 (implementation) of adding aten::layer_norm. Use mojo-gpu-kernels skill."

3. **Ask for Templates**: Request specific templates when starting from scratch
   - "Give me the reduction template from mojo-gpu-kernels skill for implementing sum"

4. **Request Comparisons**: Ask about trade-offs between approaches
   - "Use mojo-gpu-kernels skill to compare warp vs. block reduction for my use case"

5. **Combine with Subagents**: Use subagents for research, then skill for implementation
   - "Based on the PyTorch research, use mojo-gpu-kernels skill to implement this"

## Common Questions

### Q: Do I need to install the skill?
A: No, it's already available in `.claude/skills/mojo-gpu-kernels/`

### Q: Can I modify the skill?
A: Yes! The skill is part of your project. Add examples, patterns, or templates as you discover them.

### Q: What if the skill doesn't cover my use case?
A: Use the existing patterns as starting points. Then consider adding your solution to the skill for future reference.

### Q: How do I know which template to use?
A: Ask! "Which template from mojo-gpu-kernels skill should I use for [your operation]?"

### Q: Can I use the skill for CPU kernels?
A: The skill focuses on GPU kernels, but many patterns (element-wise, reductions) apply to CPU with modifications.

## Advanced Usage

### Extending the Skill

Add your own patterns:

1. Document the pattern in `references/`
2. Create a template in `assets/`
3. Update `SKILL.md` with a reference
4. Update `README.md` if it's a major addition

### Project-Specific Conventions

The skill is tailored for torch-max-backend:
- Follows test-driven development
- Uses beartype for type hints
- Integrates with pytest workflow
- Uses uv for package management
- Follows pre-commit hooks

### Combining with Other Skills

The skill works well with:
- Code review skills
- Testing skills
- Performance profiling skills
- Documentation skills

Example:
```
Use mojo-gpu-kernels skill to implement the operation, then use a code review skill to check for issues.
```

## Getting Help

If you're unsure how to use the skill:

```
How do I use the mojo-gpu-kernels skill to [task]?
```

Claude will explain the best approach for your specific task.

## Summary

The mojo-gpu-kernels skill is your companion for GPU kernel development in this project. It provides:

- ✅ Patterns and best practices
- ✅ Ready-to-use templates
- ✅ MAX framework integration
- ✅ Performance guidance
- ✅ Project workflow integration

Just reference it in your prompts and it will provide targeted help for your GPU kernel development tasks!
