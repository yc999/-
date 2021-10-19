package main

import "fmt"

func A() {
	defer A1()
	defer A2()
	panic("panicA")
}

func A1() {
	fmt.Println("A1")
}

func A2() {
	defer B1()
	panic("panicA2")
}
func B1() {
	// p := recover()
	// fmt.Println(p)
}
func main() {
	// A()
	a := []int{1, 2, 3, 4}
	b := append(a[:3], 5)

	b = append(b, 6)
	b[0] += 1

	for i := range a {
		print(a[i])

	}
	print("\n")
	for i := range b {
		print(b[i])

	}
}
