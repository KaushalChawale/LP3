import java.util.Scanner;


class Pg1_b{

	static int Fibonacci(int n){

		if(n <= 1){
			return n;
		}

		return Fibonacci(n-1) + Fibonacci(n-2);

	}

	public static void main(String[] args) {
		Scanner sc = new Scanner(System.in);
		System.out.println("Enter the length of fibonacci series: ");
		int N = sc.nextInt();
		System.out.println("The fibonacci series is:");
		for(int i=0; i < N; i++){
			System.out.print(Fibonacci(i)+" ");
		}
	}

}

// The complexity of the above method

// Time Complexity: O(2^N)  
// Auxiliary Space: O(n)