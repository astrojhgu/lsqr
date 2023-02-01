use lsqr::lsqr;
use ndarray::{ArrayView1, array};

fn main() {
    let a=array![[1.0, 1.0],
    [2.0, 1.0],
    [3.0, 1.0], 
    [4.0, 1.0],
    [5.0, 1.0],
    ];
    let b=array![2.0, 3.0, 4.0, 5.0, 6.0];

    let aprod=|x: ArrayView1<f64>|{
        assert_eq!(x.len(), 2);
        a.dot(&x)
    };

    let atprod=|x: ArrayView1<f64>|{
        assert_eq!(x.len(), b.len());
        a.t().dot(&x)
    };

    let mut x=array![2.0, 3.0];
    //println!("{}", aprod(x.view())) ;
    lsqr(x.view_mut(), &aprod, &atprod, b.view(), 1e-12, 1e-12, 1e99, 30);
    println!("{}", x);
}
