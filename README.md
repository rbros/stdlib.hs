# Minimal alternative prelude for Haskell

`stdlib` is a minimal alternative prelude for Haskell used in Reichert
Brothers projects such as [Assertible](https://assertible.com),
[SimplyRETS](https://simplyrets.com), &
[Identibyte](https://identibyte.com).

Ensure you have the `{-# LANGUAGE NoImplicitPrelude #-}` extension added
to your cabal file or at the top of each module in your project that
imports `StdLib`.

The initial implementation of `stdlib` was heavily inspired by Stephen
Diehl's [protolude](https://github.com/sdiehl/protolude)
