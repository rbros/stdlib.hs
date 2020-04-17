{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# OPTIONS_GHC -fno-warn-unused-imports #-}

-- | A custom prelude
--
-- $license

module StdLib (

      identity
    , (&)
    , uncons
    , applyN
    , print
    , io

    , LText
    , LByteString

    , CIText
    , CIByteString

      -- * List

    , (L.\\)
    , head
    , nubOrd
    , nubOn
    , sortOn
    , count
    , chunksOf

      -- * Show

    , Print(..)
    , putText
    , putLText
    , tshow
    , bshow
    , blshow

      -- * Bool

    , bool
    , whenM
    , unlessM
    , ifM
    , guardM
    , while
    , whileM

      -- * Debug

    , undefined
    , error
    , trace
    , traceM
    , traceIO
    , traceShow
    , traceShowM
    , notImplemented

      -- * Monad

    , Monad(..)
    , MonadPlus(..)

    , (=<<)
    , (>=>)
    , (<=<)
    , forever

    , join
    , mfilter
    , filterM
    , mapAndUnzipM
    , zipWithM
    , zipWithM_
    , foldM
    , foldM_
    , replicateM
    , replicateM_
    , concatMapM
    , concatForM
    , whenJust
    , unlessNull

    , guard
    , when
    , unless

    , liftM
    , liftM2
    , liftM3
    , liftM4
    , liftM5
    , liftM'
    , liftM2'
    , ap

    , (<$!>)

      -- * Applicative

    , orAlt
    , orEmpty
    , eitherA

      -- * Either

    , maybeToLeft
    , maybeToRight
    , leftToMaybe
    , rightToMaybe
    , eitherToMaybe

      -- * Text

    , txtpack
    , txtunpack
    , ltxtpack
    , ltxtunpack
    , txt2ltxt
    , txt2bs
    , bs2txt
    , lbs2txt
    , ltxt2lbs
    , bs2ltxt
    , lbs2bs
    , bs2lbs

      -- * Set

    , setFromList

      -- * Map

    , mapFromList

      -- * Case-insensitive text

    , ci
    , ciOriginal

      -- * Time

    , Milliseconds
    , utcTimeToEpochTime

      -- * Async

    , async_

      -- * Re-exports

    , (P.$)
    , (P.$!)
    , P.minBound
    , P.maxBound

    , module X
    ) where

import qualified Control.Monad.IO.Class as IO (liftIO)
import qualified Data.ByteString as X (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Char8 as BS8
import qualified Data.ByteString.Lazy
import qualified Data.ByteString.Lazy as BL
import qualified Data.ByteString.Lazy.Char8 as BL8
import qualified Data.CaseInsensitive as CI
import qualified Data.List as L
import qualified Data.Map as Map
import qualified Data.Set as Set
import qualified Data.Text as T
import qualified Data.Text.Encoding as T
import qualified Data.Text.IO as T
import qualified Data.Text.Lazy
import qualified Data.Text.Lazy as TL
import qualified Data.Text.Lazy.Encoding as TL
import qualified Data.Text.Lazy.IO as TL
import qualified Debug.Trace as T
import qualified Prelude as P
import qualified Prelude

import           Control.Applicative as X (Alternative (..), Applicative (..),
                                           Const (..), ZipList (..), liftA,
                                           liftA2, liftA3, optional, (<**>))
import           Control.Concurrent as X hiding (throwTo)
import           Control.Concurrent.Async as Async
import           Control.Monad hiding ((<$!>))
import           Control.Monad.Except as X (Except, ExceptT, MonadError,
                                            catchError, runExcept, runExceptT,
                                            throwError)
import           Control.Monad.IO.Class as X (MonadIO)
import           Control.Monad.Reader as X (MonadReader, Reader, ReaderT, ask,
                                            asks, local, runReader, runReaderT)
import           Control.Monad.ST as X
import           Control.Monad.State as X (MonadState, State, StateT,
                                           evalStateT, execState, execStateT,
                                           get, gets, modify, put, runStateT,
                                           withState)
import           Control.Monad.Trans as X (MonadTrans (..), lift)
import           Control.Monad.Trans.Maybe as X (runMaybeT)
import           Data.Bits as X
import           Data.Bool as X hiding (bool)
import           Data.ByteString as X (ByteString)
import           Data.Char as X (Char, isSpace)
import           Data.Complex as X
import           Data.Either as X
import           Data.Eq as X
import           Data.Foldable as X hiding (foldl1, foldr1, maximum, maximumBy,
                                     minimum, minimumBy)
import           Data.Function as X (const, fix, flip, on, (.))
import           Data.Functor as X (Functor (..), void, ($>), (<$>))
import           Data.Functor.Identity as X
import           Data.Int as X
import           Data.IntMap as X (IntMap)
import           Data.IntSet as X (IntSet)
import           Data.List as X (break, drop, filter, intercalate, isPrefixOf,
                                 replicate, reverse, sortBy, splitAt, take)
import           Data.List.NonEmpty as X (NonEmpty (..), nonEmpty)
import           Data.Map as X (Map)
import           Data.Maybe as X hiding (fromJust)
import           Data.Monoid as X hiding ((<>))
import           Data.Ord as X
import           Data.Semigroup as X hiding (First, Last, getFirst, getLast)
import           Data.Sequence as X (Seq)
import           Data.Set as X (Set)
import           Data.Text as X (Text)
import           Data.Time as X
import           Data.Time.Clock.POSIX
import           Data.Traversable as X
import           Data.Tuple as X
import           Data.Word as X
import           GHC.Enum as X (Enum (..))
import           GHC.Exts as X (Constraint, FunPtr, Ptr, the)
import           GHC.Float as X hiding (log)
import           GHC.Int as X
import           GHC.IO as X (IO)
import           GHC.Num as X
import           GHC.Real as X
import           GHC.Show as X
import           GHC.Stack
import           Numeric as X (showHex)
import           System.Directory as X (createDirectoryIfMissing, doesFileExist,
                                        getCurrentDirectory,
                                        setCurrentDirectory)
import           System.Environment as X (getArgs, getEnvironment, lookupEnv,
                                          setEnv, unsetEnv)
import           System.Exit as X
import           System.FilePath as X (FilePath, takeBaseName, (</>))
import           System.IO as X (BufferMode (..), Handle, hClose, hFlush,
                                 hSetBinaryMode, hWaitForInput)
import           System.Process as X (readProcess, readProcessWithExitCode)
import           Text.Printf as X (PrintfArg, hPrintf, printf)
import           Text.Read as X (Read, readEither, readMaybe, reads)

type LText = Data.Text.Lazy.Text

type LByteString = Data.ByteString.Lazy.ByteString

type CIText = CI.CI Text

type CIByteString = CI.CI ByteString

infixl 1 &

(&) :: a -> (a -> b) -> b
x & f = f x

identity :: a -> a
identity x = x

uncons :: [a] -> Maybe (a, [a])
uncons []     = Nothing
uncons (x:xs) = Just (x, xs)

applyN :: Int -> (a -> a) -> a -> a
applyN n f = X.foldr (.) identity (X.replicate n f)

print :: (MonadIO m, Show a) => a -> m ()
print = io . putStrLn . tshow

io :: MonadIO m => IO a -> m a
io = IO.liftIO

-- * List

head :: Foldable f => f a -> Maybe a
head = foldr (\x _ -> return x) Nothing

sortOn :: Ord o => (a -> o) -> [a] -> [a]
sortOn = sortBy . comparing

-- O(n * log n)
nubOrd :: Ord a => [a] -> [a]
nubOrd l = go Set.empty l
  where
    go _ []     = []
    go s (x:xs) =
      if x `Set.member` s
      then go s xs
      else x : go (Set.insert x s) xs

nubOn :: Eq b => (a -> b) -> [a] -> [a]
nubOn f = fmap snd
        . L.nubBy ((==) `on` fst)
        . fmap (\x -> let y = f x in y `Prelude.seq` (y, x))

count :: Eq a => a -> [a] -> Int
count x = length . filter (x ==)

chunksOf :: Int -> [a] -> [[a]]
chunksOf _ [] = []
chunksOf n xs = take n xs : chunksOf n (drop n xs)

-- * Show

class Print a where
    putStr :: MonadIO m => a -> m ()
    putStrLn :: MonadIO m => a -> m ()

instance Print Text where
    putStr = io . T.putStr
    putStrLn = io . T.putStrLn

instance Print LText where
    putStr = io . TL.putStr
    putStrLn = io . TL.putStrLn

instance Print BS.ByteString where
    putStr = io . BS.putStr
    putStrLn = io . BS8.putStrLn

instance Print LByteString where
    putStr = io . BL.putStr
    putStrLn = io . BL8.putStrLn

instance Print [Char] where
    putStr = io . Prelude.putStr
    putStrLn = io . Prelude.putStrLn

putText :: MonadIO m => Text -> m ()
putText = putStrLn
{-# SPECIALIZE putText :: Text -> IO () #-}

putLText :: MonadIO m => LText -> m ()
putLText = putStrLn
{-# SPECIALIZE putLText :: LText -> IO () #-}

tshow :: Show a => a -> Text
tshow = T.pack . show

{-# WARNING bshow "'bshow' remains in code" #-}
bshow :: Show a => a -> ByteString
bshow = BS8.pack . show

{-# WARNING blshow "'blshow' remains in code" #-}
blshow :: Show a => a -> LByteString
blshow = BL8.pack . show

-- * Bool

bool :: a -> a -> Bool -> a
bool f t p = if p then t else f

whenM :: Monad m => m Bool -> m () -> m ()
whenM p m = p >>= flip when m

unlessM :: Monad m => m Bool -> m () -> m ()
unlessM p m = p >>= flip unless m

ifM :: Monad m => m Bool -> m a -> m a -> m a
ifM p x y = p >>= \b -> if b then x else y

guardM :: MonadPlus m => m Bool -> m ()
guardM f = guard =<< f

while :: Monad m => m Bool -> m a -> m ()
while cond action = do
    c <- cond
    when c P.$ do
        _ <- action
        while cond action

-- | While loop. run 'f' while 'p'
whileM :: (MonadPlus m1, Monad m) => (a -> m Bool) -> m a -> m (m1 a)
whileM p f = go
  where
    go = do
      x <- f
      r <- p x
      if r then (return x `mplus`) `liftM` go else return mzero

{-# WARNING undefined "'undefined' remains in code" #-}
undefined :: HasCallStack => a
undefined = P.undefined

{-# WARNING error "'error' remains in code" #-}
error :: HasCallStack => P.String -> a
error = P.error

{-# WARNING trace "'trace' remains in code" #-}
trace :: P.String -> a -> a
trace = T.trace

{-# WARNING traceShow "'traceShow' remains in code" #-}
traceShow :: Show a => a -> a
traceShow a = T.trace (P.show a) a

{-# WARNING traceShowM "'traceShowM' remains in code" #-}
traceShowM :: (Show a, Monad m) => a -> m ()
traceShowM a = T.traceM (P.show a)

{-# WARNING traceM "'traceM' remains in code" #-}
traceM :: Monad m => P.String -> m ()
traceM = T.traceM

{-# WARNING traceIO "'traceIO' remains in code" #-}
traceIO :: P.String -> IO ()
traceIO = T.traceIO

{-# WARNING notImplemented "'notImplemented' remains in code" #-}
notImplemented :: a
notImplemented = P.error "Not implemented"

-- * Monad

concatMapM :: Monad m => (a -> m [b]) -> [a] -> m [b]
concatMapM f xs = liftM concat (mapM f xs)

whenJust :: Monad m => Maybe a -> m () -> m ()
whenJust = when . isJust

unlessNull :: Monad m => [a] -> m () -> m ()
unlessNull = unless . L.null

concatForM :: (Monad m, Traversable t) => t a -> (a -> m [b]) -> m [b]
concatForM ls m = fmap concat (forM ls m)

liftM' :: Monad m => (a -> b) -> m a -> m b
liftM' = (<$!>)
{-# INLINE liftM' #-}

liftM2' :: Monad m => (a -> b -> c) -> m a -> m b -> m c
liftM2' f a b = do
    x <- a
    y <- b
    let z = f x y
    z `Prelude.seq` return z
{-# INLINE liftM2' #-}

(<$!>) :: Monad m => (a -> b) -> m a -> m b
f <$!> m = do
    x <- m
    let z = f x
    z `Prelude.seq` return z
{-# INLINE (<$!>) #-}

-- * Applicative

orAlt :: (Alternative f, Monoid a) => f a -> f a
orAlt f = f <|> pure mempty

orEmpty :: Alternative f => Bool -> a -> f a
orEmpty b a = if b then pure a else empty

eitherA :: Alternative f => f a -> f b -> f (Either a b)
eitherA a b = (Left <$> a) <|> (Right <$> b)

-- * Either

leftToMaybe :: Either l r -> Maybe l
leftToMaybe = either Just (const Nothing)

rightToMaybe :: Either l r -> Maybe r
rightToMaybe = either (const Nothing) Just

maybeToRight :: l -> Maybe r -> Either l r
maybeToRight l = maybe (Left l) Right

maybeToLeft :: r -> Maybe l -> Either l r
maybeToLeft r = maybe (Right r) Left

eitherToMaybe :: Either a b -> Maybe b
eitherToMaybe (Left _)  = Nothing
eitherToMaybe (Right r) = Just r

-- * Text / ByteString

txtpack :: Prelude.String -> Text
txtpack = T.pack

txtunpack :: Text -> Prelude.String
txtunpack = T.unpack

ltxtpack :: Prelude.String -> LText
ltxtpack = TL.pack

ltxtunpack :: LText -> Prelude.String
ltxtunpack = TL.unpack

txt2ltxt :: Text -> LText
txt2ltxt = TL.fromStrict

txt2bs :: Text -> ByteString
txt2bs = T.encodeUtf8

bs2txt :: ByteString -> Text
bs2txt = T.decodeUtf8

lbs2txt :: LByteString -> Text
lbs2txt = TL.toStrict . TL.decodeUtf8

ltxt2lbs :: LText -> LByteString
ltxt2lbs = TL.encodeUtf8

bs2ltxt :: LByteString -> LText
bs2ltxt = TL.decodeUtf8

lbs2bs :: LByteString -> ByteString
lbs2bs = BL.toStrict

bs2lbs :: ByteString -> LByteString
bs2lbs = BL.fromChunks . (:[])

-- * Set

setFromList :: Ord a => [a] -> Set a
setFromList = Set.fromList

-- * Map

mapFromList :: Ord a => [(a, b)] -> Map a b
mapFromList = Map.fromList

-- * Case-insensitive

ci :: CI.FoldCase s => s -> CI.CI s
ci = CI.mk

ciOriginal :: CI.CI a -> a
ciOriginal = CI.original

-- * Time

type Milliseconds = Double

utcTimeToEpochTime :: UTCTime -> Integer
utcTimeToEpochTime = round . utcTimeToPOSIXSeconds

-- * Async

-- | Just like 'Async.async' but ignores the return value 'Async'
--
-- An 'Async' is a Haskell RTS thread that writes to a TMVar when
-- finished.
--
-- 'cancel' will send the Haskell thread a @SIGKILL@ signal but you don't
-- need to explicitly kill the thread. If the 'Async' can be garbage
-- collected, the thread will still run to completion and then be
-- properly cleaned up.
--
-- However, if there is an exception in the 'Async', then an exception
-- won't be propagate to the "parent" thread (unless 'wait' is somehow
-- used)
async_ :: IO a -> IO ()
async_ = void . Async.async
