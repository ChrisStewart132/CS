public interface Iterator<T> {
    T first();          // Moves the iterator to the first element of the collection and returns it.
    T next();           // Moves the iterator to the next element of the collection and returns it.
    boolean isDone();   // Checks if the iterator has reached the end of the collection.
    T currentItem();    // Returns the current element pointed to by the iterator.
}

