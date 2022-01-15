use rgb::ComponentMap;
use imagequant::*;


#[test]
fn remap_bg6() {
    let fg1 = lodepng::decode32_file("tests/frame-6-a.png").unwrap();
    let bg1 = lodepng::decode32_file("tests/frame-6-bg.png").unwrap();
    let pal = lodepng::decode32_file("tests/frame-6-pal.png").unwrap();

    let mut attr = new();
    let mut fg = attr.new_image_stride(&fg1.buffer, fg1.width, fg1.height, fg1.width, 0.).unwrap();
    let bg = attr.new_image_stride(&bg1.buffer, bg1.width, bg1.height, bg1.width, 0.).unwrap();
    fg.set_background(bg).unwrap();
    for c in &pal.buffer {
        fg.add_fixed_color(*c).unwrap();
    }
    attr.set_max_colors(pal.buffer.len() as _).unwrap();
    let mut res = attr.quantize(&mut fg).unwrap();
    res.set_dithering_level(1.).unwrap();
    let (pal, idx) = res.remapped(&mut fg).unwrap();

    let buf: Vec<_> = idx.iter().zip(bg1.buffer.iter()).map(|(px, bg)| {
        let palpx = pal[*px as usize];
        if palpx.a > 0 {
            palpx
        } else {
            *bg
        }
    }).collect();
    lodepng::encode32_file("/tmp/testr2-r6.png", &buf, fg.width(), fg.height()).unwrap();

    assert!(idx.iter().zip(bg1.buffer.iter()).map(|(px, bg)| {
        let palpx = pal[*px as usize];
        if palpx.a > 0 {
            palpx
        } else {
            *bg
        }
    }).zip(&fg1.buffer).all(|(px, fg)| {
        let d = px.map(|c| c as i16) - fg.map(|c| c as i16);
        d.map(|c| (c as i32).pow(2) as u32);
        d.r + d.g + d.b + d.a < 120
    }));
}

#[test]
fn remap_bg7() {
    let fg1 = lodepng::decode32_file("tests/frame-7-a.png").unwrap();
    let bg1 = lodepng::decode32_file("tests/frame-7-bg.png").unwrap();
    let pal = lodepng::decode32_file("tests/frame-7-pal.png").unwrap();

    let mut attr = new();
    let mut fg = attr.new_image_stride(&fg1.buffer, fg1.width, fg1.height, fg1.width, 0.).unwrap();
    let bg = attr.new_image_stride(&bg1.buffer, bg1.width, bg1.height, bg1.width, 0.).unwrap();
    fg.set_background(bg).unwrap();
    for c in &pal.buffer {
        fg.add_fixed_color(*c).unwrap();
    }
    attr.set_max_colors(pal.buffer.len() as _).unwrap();
    let mut res = attr.quantize(&mut fg).unwrap();
    res.set_dithering_level(0.).unwrap();
    let (pal, idx) = res.remapped(&mut fg).unwrap();

    let buf: Vec<_> = idx.iter().zip(bg1.buffer.iter()).map(|(px, bg)| {
        let palpx = pal[*px as usize];
        if palpx.a > 0 {
            palpx
        } else {
            *bg
        }
    }).collect();
    lodepng::encode32_file("/tmp/testr2-r7.png", &buf, fg.width(), fg.height()).unwrap();

    assert!(idx.iter().zip(bg1.buffer.iter()).map(|(px, bg)| {
        let palpx = pal[*px as usize];
        if palpx.a > 0 {
            palpx
        } else {
            *bg
        }
    }).zip(&fg1.buffer).all(|(px, fg)| {
        let d = px.map(|c| c as i16) - fg.map(|c| c as i16);
        d.map(|c| (c as i32).pow(2) as u32);
        d.r + d.g + d.b + d.a < 160
    }));
}
