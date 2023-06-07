import java.util.Arrays;
import java.util.Scanner;


public class Main{
    public static void main(String[] args){
        System.out.println("Design Patterns\n");


        // Iterator pattern
        // consumer class only cares about the full iteration of a collection
        // the implementation of the Iterator interface dictates the method of traversal (inOrder, dfs, bfs etc.)
        Integer[] data = {1,2,3,4};
        ArrayIterator arrayIterator = new ArrayIterator<Integer>(data);

        Integer[] iteratedData = new Integer[data.length];
        for(int i = 0; !arrayIterator.isDone(); i++) {
            iteratedData[i] = (Integer)arrayIterator.next();
        }
        System.out.println(Arrays.toString(data));
        System.out.println(Arrays.toString(iteratedData));


        // Singleton pattern
        // cannot be constructed publically, must be statically called
        // Singleton singleton = Singleton.getInstance()
        // getInstance() creates the one and only instance or returns the already created instance

        // creating 2 reference variables to the Singleton class
        Singleton singleton1 = Singleton.getInstance();
        Singleton singleton2 = Singleton.getInstance();
        // confirm that both references point to the same object in memory
        System.out.println(singleton1 == singleton2);


        
        Scanner scanner = new Scanner(System.in);
        String input = scanner.nextLine();
        return;
    }
}