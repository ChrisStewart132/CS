public class ArrayIterator<T> implements Iterator<T> {
    T[] array;
    int i = 0;

    public ArrayIterator(T[] arr) {
        array = arr;
    }

    @Override
    public T first() {
        i = 0;
        return array[i];
    }

    @Override
    public T next() {
        return array[i++];
    }

    @Override
    public boolean isDone() {
        return i >= array.length;
    }

    @Override
    public T currentItem() {
        return array[i];
    }
}
