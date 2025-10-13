use super::*;

impl Fractal {
    pub fn nebulabrot_or_antinebulabrot(
        &mut self,
        is_antinebulabrot: bool,
        rounds: u32,
        red_iter: u32,
        green_iter: u32,
        blue_iter: u32,
        color_shift: Option<u32>,
        uniform_factor: Option<f64>,
    ) -> Vec<Vec<(u8, u8, u8)>> {
        // I could try to make this more efficient, but honestly,
        // it's not worth it, the difference between nebulabrot and buddhabrot is 1.7x.

        let red_buddha;
        let green_buddha;
        let blue_buddha;
        if is_antinebulabrot {
            self.iterations = red_iter;
            red_buddha = self.antibuddhabrot(rounds);
            self.iterations = green_iter;
            green_buddha = self.antibuddhabrot(rounds);
            self.iterations = blue_iter;
            blue_buddha = self.antibuddhabrot(rounds);
        } else {
            self.iterations = red_iter;
            red_buddha = self.buddhabrot(rounds);
            self.iterations = green_iter;
            green_buddha = self.buddhabrot(rounds);
            self.iterations = blue_iter;
            blue_buddha = self.buddhabrot(rounds);
        }

        let start = Instant::now();
        let default_color = (0, 0, 0);
        let mut color_bitmap = vec![vec![default_color; self.height as usize]; self.width as usize];
        let mut red_tmp: Vec<&u32> = red_buddha
            .iter()
            .take(self.width as usize * self.height as usize)
            .collect();
        sort(&mut red_tmp);
        let red_max_param = *red_tmp[red_tmp.len() - 100];
        let mut green_tmp: Vec<&u32> = green_buddha
            .iter()
            .take(self.width as usize * self.height as usize)
            .collect();
        sort(&mut green_tmp);
        let green_max_param = *green_tmp[green_tmp.len() - 100];
        let mut blue_tmp: Vec<&u32> = blue_buddha
            .iter()
            .take(self.width as usize * self.height as usize)
            .collect();
        sort(&mut blue_tmp);
        let blue_max_param = *blue_tmp[blue_tmp.len() - 100];
        for x in 0..self.width as usize {
            for y in 0..self.height as usize {
                let hits_red = red_buddha[x * self.height as usize + y];
                let hits_green = green_buddha[x * self.height as usize + y];
                let hits_blue = blue_buddha[x * self.height as usize + y];
                let color_red = set_color(
                    hits_red,
                    red_max_param,
                    PaletteMode::GrayScale { shift: color_shift , uniform_factor},
                )
                .0;
                let color_green = set_color(
                    hits_green,
                    green_max_param,
                    PaletteMode::GrayScale { shift: color_shift , uniform_factor},
                )
                .0;
                let color_blue = set_color(
                    hits_blue,
                    blue_max_param,
                    PaletteMode::GrayScale { shift: color_shift , uniform_factor},
                )
                .0;
                color_bitmap[x][y] = (color_red, color_green, color_blue);
            }
        }

        println!("Time taken to color nebula: {:.2?}", start.elapsed());

        color_bitmap
    }

    pub fn nebulabrot(
        &mut self,
        rounds: u32,
        red_iter: u32,
        green_iter: u32,
        blue_iter: u32,
        color_shift: Option<u32>,
        uniform_factor: Option<f64>,
    ) -> Vec<Vec<(u8, u8, u8)>> {
        self.nebulabrot_or_antinebulabrot(
            false,
            rounds,
            red_iter,
            green_iter,
            blue_iter,
            color_shift,
            uniform_factor,
        )
    }

    pub fn antinebulabrot(
        &mut self,
        rounds: u32,
        red_iter: u32,
        green_iter: u32,
        blue_iter: u32,
        color_shift: Option<u32>,
        uniform_factor: Option<f64>,
    ) -> Vec<Vec<(u8, u8, u8)>> {
        self.nebulabrot_or_antinebulabrot(
            true,
            rounds,
            red_iter,
            green_iter,
            blue_iter,
            color_shift,
            uniform_factor,
        )
    }
}
