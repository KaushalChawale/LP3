// Iterative function to generate Fibonacci series up to N terms
import java.util.Scanner;

class Pg1_a {
    
    static void Fibonacci(int N) {
        int no1 = 0; 
        int no2 = 1;
        for(int i=0;i<N;i++){
            System.out.print(no1+" ");
            int sum = no1 + no2;
            no1 = no2;
            no2 = sum;
        }
    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.println("Enter the length of Fibonacci series: ");
        int N = sc.nextInt();

        System.out.println("The Fibonacci series is: ");
        Fibonacci(N);  // Call the Fibonacci function to print the series
    }
}

// The complexity of the above method
// Time Complexity: O(N) 
// Auxiliary Space: O(1)