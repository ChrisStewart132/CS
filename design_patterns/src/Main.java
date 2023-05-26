import java.util.Scanner;


public class Main{
    public static void main(String[] args){
        System.out.println("init");

        Integer[] data = {1,2,3,4};
        ArrayIterator aI = new ArrayIterator<Integer>(data);

        while(!aI.isDone()){
            Integer i = (Integer)aI.next();
            System.out.println(Integer.toString(i));
        }

        Scanner scanner = new Scanner(System.in);
        System.out.print("Exit?");
        String input = scanner.nextLine();

        return;
    }
}