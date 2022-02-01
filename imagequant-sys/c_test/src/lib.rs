#[cfg(test)]
extern crate imagequant_sys;

#[cfg(test)]
extern "C" {
    fn run_liq_tests();
}

#[test]
fn c_test() {
    unsafe {
        run_liq_tests();
    }
}
