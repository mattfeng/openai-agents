# Environments


## Enduro
### About
The National Enduro

See `ale/enduro_manual.pdf` for more information
about the Atari game itself.

### Observations

### Actions

```python
print(M.env.unwrapped._action_set())
>>> [ 0  1  3  4  5  8  9 11 12]
```

NOP
Fire
Right
Left
Down
Down-Right
Down-Left
Right-Fire
Left-Fire