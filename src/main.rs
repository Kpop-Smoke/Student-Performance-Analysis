use std::env;
use std::error::Error;
use std::fs::File;
use std::path::Path;
use csv::Reader;
use ndarray::Array2;
use linfa::Dataset;
use linfa_clustering::KMeans;
use linfa::prelude::*;
use plotters::prelude::*;
use serde::Deserialize;

// Define colors
const PURPLE: RGBColor = RGBColor(128, 0, 128);
const ORANGE: RGBColor = RGBColor(255, 165, 0);
const CYAN: RGBColor = RGBColor(0, 255, 255);

#[derive(Debug, Deserialize)]
pub struct Student {
    pub study_hours_per_day: f32,
    pub sleep_hours: f32,
    pub mental_health_rating: u8,
    pub exam_score: f32,
    pub social_media_hours: f32,
    pub netflix_hours: f32,
    pub attendance_percentage: f32,
    pub exercise_frequency: String,
}

// Load students from the CSV file
pub fn load_students<P: AsRef<Path>>(filename: P) -> Result<Vec<Student>, Box<dyn Error>> {
    let file = File::open(filename)?; 
    let mut rdr = Reader::from_reader(file); 
    let mut students = Vec::new(); 

    for result in rdr.deserialize() {
        let record: Student = result?; 
        students.push(record); 
    }

    Ok(students)
}

// Helper function to calculate correlation
pub fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_x2: f64 = x.iter().map(|a| a * a).sum();
    let sum_y2: f64 = y.iter().map(|a| a * a).sum();
    
    let numerator = n * sum_xy - sum_x * sum_y;
    let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 { 0.0 } else { numerator / denominator }
}

// Function to normalize data between 0 and 1
pub fn normalize(data: &[f64]) -> Vec<f64> {
    if data.is_empty() {
        return vec![];
    }
    let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = max - min;
    if range == 0.0 {
        return vec![0.0; data.len()];
    }
    data.iter().map(|&x| (x - min) / range).collect()
}

// Analysis functions
pub fn analyze_study_hours_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    let mut study_hours: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    
    for student in students {
        study_hours.push(student.study_hours_per_day as f64);
        scores.push(student.exam_score as f64);
    }
    
    let correlation = calculate_correlation(&study_hours, &scores);
    println!("Correlation between study hours and exam scores: {:.3}", correlation);
    
    // Create scatter plot with enhanced styling
    let root = BitMapBackend::new("study_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Study Hours vs Exam Scores", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..10f32, 0f32..100f32)?;
    
    // Configure mesh with better styling
    chart.configure_mesh()
        .x_desc("Study Hours per Day")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;
    
    // Draw scatter points with label
    chart.draw_series(
        study_hours.iter()
            .zip(scores.iter())
            .map(|(x, y)| Circle::new((*x as f32, *y as f32), 4, BLUE.filled())),
    )?.label("Student Data Points")
      .legend(move |(x, y)| Circle::new((x, y), 4, BLUE.filled()));
    
    // Add correlation coefficient as annotation
    let correlation_text = format!("Correlation: {:.3}", correlation);
    root.draw(&Text::new(
        correlation_text,
        (50, 50),
        ("sans-serif", 20).into_font(),
    ))?;
    
    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn analyze_sleep_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    let mut sleep_hours: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    
    for student in students {
        sleep_hours.push(student.sleep_hours as f64);
        scores.push(student.exam_score as f64);
    }
    
    let correlation = calculate_correlation(&sleep_hours, &scores);
    println!("Correlation between sleep hours and exam scores: {:.3}", correlation);
    
    // Create scatter plot with enhanced styling
    let root = BitMapBackend::new("sleep_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Sleep Hours vs Exam Scores", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..12f32, 0f32..100f32)?;
    
    // Configure mesh with better styling
    chart.configure_mesh()
        .x_desc("Sleep Hours per Day")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;
    
    // Draw scatter points with label
    chart.draw_series(
        sleep_hours.iter()
            .zip(scores.iter())
            .map(|(x, y)| Circle::new((*x as f32, *y as f32), 4, GREEN.filled())),
    )?.label("Student Data Points")
      .legend(move |(x, y)| Circle::new((x, y), 4, GREEN.filled()));
    
    // Add correlation coefficient as annotation
    let correlation_text = format!("Correlation: {:.3}", correlation);
    root.draw(&Text::new(
        correlation_text,
        (50, 50),
        ("sans-serif", 20).into_font(),
    ))?;
    
    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn analyze_mental_health_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    let mut mental_health: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    
    for student in students {
        mental_health.push(student.mental_health_rating as f64);
        scores.push(student.exam_score as f64);
    }
    
    let correlation = calculate_correlation(&mental_health, &scores);
    println!("Correlation between mental health rating and exam scores: {:.3}", correlation);
    
    // Create scatter plot with enhanced styling
    let root = BitMapBackend::new("mental_health_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Mental Health Rating vs Exam Scores", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..10f32, 0f32..100f32)?;
    
    // Configure mesh with better styling
    chart.configure_mesh()
        .x_desc("Mental Health Rating (1-10)")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;
    
    // Draw scatter points with label
    chart.draw_series(
        mental_health.iter()
            .zip(scores.iter())
            .map(|(x, y)| Circle::new((*x as f32, *y as f32), 4, PURPLE.filled())),
    )?.label("Student Data Points")
      .legend(move |(x, y)| Circle::new((x, y), 4, PURPLE.filled()));
    
    // Add correlation coefficient as annotation
    let correlation_text = format!("Correlation: {:.3}", correlation);
    root.draw(&Text::new(
        correlation_text,
        (50, 50),
        ("sans-serif", 20).into_font(),
    ))?;
    
    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn analyze_social_media_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    let mut social_media: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    
    for student in students {
        social_media.push(student.social_media_hours as f64);
        scores.push(student.exam_score as f64);
    }
    
    let correlation = calculate_correlation(&social_media, &scores);
    println!("Correlation between social media hours and exam scores: {:.3}", correlation);
    
    // Create scatter plot with enhanced styling
    let root = BitMapBackend::new("social_media_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Social Media Hours vs Exam Scores", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..10f32, 0f32..100f32)?;
    
    chart.configure_mesh()
        .x_desc("Social Media Hours per Day")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;
    
    chart.draw_series(
        social_media.iter()
            .zip(scores.iter())
            .map(|(x, y)| Circle::new((*x as f32, *y as f32), 4, ORANGE.filled())),
    )?.label("Student Data Points")
      .legend(move |(x, y)| Circle::new((x, y), 4, ORANGE.filled()));
    
    let correlation_text = format!("Correlation: {:.3}", correlation);
    root.draw(&Text::new(
        correlation_text,
        (50, 50),
        ("sans-serif", 20).into_font(),
    ))?;
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn analyze_attendance_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    let mut attendance: Vec<f64> = Vec::new();
    let mut scores: Vec<f64> = Vec::new();
    
    for student in students {
        attendance.push(student.attendance_percentage as f64);
        scores.push(student.exam_score as f64);
    }
    
    let correlation = calculate_correlation(&attendance, &scores);
    println!("Correlation between attendance percentage and exam scores: {:.3}", correlation);
    
    // Create scatter plot with enhanced styling
    let root = BitMapBackend::new("attendance_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Attendance Percentage vs Exam Scores", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..100f32, 0f32..100f32)?;
    
    chart.configure_mesh()
        .x_desc("Attendance Percentage")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;
    
    chart.draw_series(
        attendance.iter()
            .zip(scores.iter())
            .map(|(x, y)| Circle::new((*x as f32, *y as f32), 4, CYAN.filled())),
    )?.label("Student Data Points")
      .legend(move |(x, y)| Circle::new((x, y), 4, CYAN.filled()));
    
    let correlation_text = format!("Correlation: {:.3}", correlation);
    root.draw(&Text::new(
        correlation_text,
        (50, 50),
        ("sans-serif", 20).into_font(),
    ))?;
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn analyze_exercise_vs_scores(students: &[Student]) -> Result<(), Box<dyn Error>> {
    // Group students by exercise frequency and calculate average scores
    let mut exercise_groups = std::collections::HashMap::new();
    let mut group_counts = std::collections::HashMap::new();
    
    for student in students {
        let entry = exercise_groups.entry(&student.exercise_frequency).or_insert(0.0);
        *entry += student.exam_score as f64;
        *group_counts.entry(&student.exercise_frequency).or_insert(0) += 1;
    }
    
    // Calculate averages
    let mut exercise_data: Vec<(String, f64)> = Vec::new();
    for (frequency, total) in exercise_groups {
        let count = group_counts[frequency] as f64;
        exercise_data.push((frequency.to_string(), total / count));
    }
    
    // Sort by exercise frequency
    exercise_data.sort_by(|a, b| a.0.cmp(&b.0));
    
    // Create bar chart
    let root = BitMapBackend::new("exercise_vs_scores.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;
    
    let mut chart = ChartBuilder::on(&root)
        .caption("Exercise Frequency vs Average Exam Score", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0f32..(exercise_data.len() as f32),
            0f32..100f32,
        )?;
    
    chart.configure_mesh()
        .x_desc("Exercise Frequency")
        .y_desc("Average Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .x_labels(exercise_data.len())
        .x_label_formatter(&|x| {
            let idx = *x as usize;
            if idx < exercise_data.len() {
                exercise_data[idx].0.clone()
            } else {
                String::new()
            }
        })
        .draw()?;
    
    chart.draw_series(
        exercise_data.iter().enumerate().map(|(i, (_, score))| {
            Rectangle::new(
                [(i as f32, 0f32), (i as f32 + 0.8, *score as f32)],
                BLUE.filled(),
            )
        }),
    )?.label("Average Score")
      .legend(move |(x, y)| Rectangle::new([(x, y - 5), (x + 20, y + 5)], BLUE.filled()));
    
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    Ok(())
}

pub fn perform_kmeans_analysis(students: &[Student]) -> Result<(), Box<dyn Error>> {
    // Prepare data for clustering
    let mut data = Vec::new();
    for student in students {
        data.push(vec![
            student.study_hours_per_day as f64,
            student.sleep_hours as f64,
            student.exam_score as f64,
        ]);
    }

    // Convert to ndarray format and normalize
    let n_samples = data.len();
    let n_features = 3;
    let mut array_data = Array2::zeros((n_samples, n_features));
    for (i, row) in data.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            array_data[[i, j]] = val;
        }
    }

    // Normalize each feature
    for i in 0..n_features {
        let col = array_data.column(i).to_vec();
        let normalized = normalize(&col);
        for (j, val) in normalized.iter().enumerate() {
            array_data[[j, i]] = *val;
        }
    }

    // Create dataset
    let dataset = Dataset::from(array_data);

    // Perform k-means clustering
    let n_clusters = 3;
    let kmeans = KMeans::params(n_clusters)
        .max_n_iterations(100)
        .tolerance(1e-5)
        .fit(&dataset)?;

    let predictions = kmeans.predict(&dataset);

    // Visualize clusters with enhanced styling
    let root = BitMapBackend::new("kmeans_clusters.png", (1000, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Student Performance Clusters", ("sans-serif", 40).into_font())
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0f32..10f32, 0f32..100f32)?;

    // Configure mesh with better styling
    chart.configure_mesh()
        .x_desc("Study Hours per Day")
        .y_desc("Exam Score")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .draw()?;

    // Different colors for different clusters with labels
    let colors = [&BLUE, &RED, &GREEN];
    let labels = ["Average Performers", "Lower Performers", "High Performers"];
    
    for cluster_id in 0..n_clusters {
        let cluster_points: Vec<(f32, f32)> = predictions
            .iter()
            .zip(students.iter())
            .filter(|(&pred, _)| pred == cluster_id)
            .map(|(_, student)| (student.study_hours_per_day, student.exam_score))
            .collect();

        chart.draw_series(
            cluster_points.iter()
                .map(|&(x, y)| Circle::new((x, y), 4, colors[cluster_id as usize].filled())),
        )?.label(labels[cluster_id as usize])
          .legend(move |(x, y)| Circle::new((x, y), 4, colors[cluster_id as usize].filled()));
    }

    // Add legend
    chart.configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;

    // Print cluster statistics
    for cluster_id in 0..n_clusters {
        let cluster_students: Vec<&Student> = predictions
            .iter()
            .zip(students.iter())
            .filter(|(&pred, _)| pred == cluster_id)
            .map(|(_, student)| student)
            .collect();

        let avg_score: f32 = cluster_students.iter().map(|s| s.exam_score).sum::<f32>() / cluster_students.len() as f32;
        let avg_study: f32 = cluster_students.iter().map(|s| s.study_hours_per_day).sum::<f32>() / cluster_students.len() as f32;
        
        println!("\nCluster {} Statistics:", cluster_id);
        println!("Number of students: {}", cluster_students.len());
        println!("Average exam score: {:.2}", avg_score);
        println!("Average study hours: {:.2}", avg_study);
    }

    Ok(())
}

pub fn perform_linear_regression(students: &[Student]) -> Result<(), Box<dyn Error>> {
    // Prepare data for linear regression
    let study_hours: Vec<f64> = students.iter()
        .map(|s| s.study_hours_per_day as f64)
        .collect();
    let exam_scores: Vec<f64> = students.iter()
        .map(|s| s.exam_score as f64)
        .collect();

    // Calculate means
    let mean_study = study_hours.iter().sum::<f64>() / study_hours.len() as f64;
    let mean_score = exam_scores.iter().sum::<f64>() / exam_scores.len() as f64;

    // Calculate coefficient (slope)
    let numerator: f64 = study_hours.iter().zip(exam_scores.iter())
        .map(|(&x, &y)| (x - mean_study) * (y - mean_score))
        .sum();
    let denominator: f64 = study_hours.iter()
        .map(|&x| (x - mean_study).powi(2))
        .sum();
    let coefficient = numerator / denominator;

    // Calculate intercept
    let intercept = mean_score - coefficient * mean_study;

    println!("\nLinear Regression Results:");
    println!("Coefficient (study hours): {:.3}", coefficient);
    println!("Intercept: {:.3}", intercept);

    // Calculate R-squared
    let predictions: Vec<f64> = study_hours.iter()
        .map(|&x| coefficient * x + intercept)
        .collect();
    
    let ss_tot: f64 = exam_scores.iter()
        .map(|&y| (y - mean_score).powi(2))
        .sum();
    let ss_res: f64 = exam_scores.iter().zip(predictions.iter())
        .map(|(&y, &y_pred)| (y - y_pred).powi(2))
        .sum();
    let r_squared = 1.0 - (ss_res / ss_tot);
    println!("R-squared: {:.3}", r_squared);

    // Create scatter plot with regression line
    let root = BitMapBackend::new("regression_line.png", (800, 600))
        .into_drawing_area();
    root.fill(&WHITE)?;

    let x_range = 0.0..8.0;
    let y_range = 0.0..100.0;

    let mut chart = ChartBuilder::on(&root)
        .caption("Study Hours vs Exam Scores (with Regression Line)", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_range, y_range)?;

    chart.configure_mesh()
        .x_desc("Study Hours per Day")
        .y_desc("Exam Score")
        .draw()?;

    // Plot scatter points
    chart.draw_series(
        students.iter().map(|s| {
            Circle::new(
                (s.study_hours_per_day as f64, s.exam_score as f64),
                3,
                BLUE.mix(0.3).filled(),
            )
        }),
    )?;

    // Plot regression line
    chart.draw_series(LineSeries::new(
        (0..80).map(|i| {
            let x = i as f64 * 0.1;
            (x, coefficient * x + intercept)
        }),
        &RED,
    ))?;

    // Add correlation coefficient to the plot
    let correlation = calculate_correlation(&study_hours, &exam_scores);
    
    chart.draw_series(std::iter::once(Text::new(
        format!("Correlation: {:.3}", correlation),
        (0.5, 90.0),
        ("sans-serif", 20).into_font(),
    )))?;

    root.present()?;

    Ok(())
}

fn main() {
    // Print the current working directory
    match env::current_dir() {
        Ok(path) => println!("Current directory: {}", path.display()),
        Err(e) => println!("Error getting current directory: {}", e),
    }

    // Set the path to your CSV file
    let filename = "student_habits_performance.csv";

    println!("Attempting to load students from: {}", filename);
    
    match load_students(filename) {
        Ok(students) => {
            println!("Successfully loaded {} students", students.len());
            
            println!("\nPerforming Machine Learning Analysis...");
            
            if let Err(e) = perform_kmeans_analysis(&students) {
                eprintln!("Error in K-means analysis: {}", e);
            }
            
            if let Err(e) = perform_linear_regression(&students) {
                eprintln!("Error in linear regression analysis: {}", e);
            }
            
            // Perform analysis
            println!("\nAnalyzing trends in the dataset...");
            
            if let Err(e) = analyze_study_hours_vs_scores(&students) {
                eprintln!("Error analyzing study hours vs scores: {}", e);
            }
            
            if let Err(e) = analyze_sleep_vs_scores(&students) {
                eprintln!("Error analyzing sleep vs scores: {}", e);
            }
            
            if let Err(e) = analyze_mental_health_vs_scores(&students) {
                eprintln!("Error analyzing mental health vs scores: {}", e);
            }

            if let Err(e) = analyze_social_media_vs_scores(&students) {
                eprintln!("Error analyzing social media vs scores: {}", e);
            }

            if let Err(e) = analyze_attendance_vs_scores(&students) {
                eprintln!("Error analyzing attendance vs scores: {}", e);
            }

            if let Err(e) = analyze_exercise_vs_scores(&students) {
                eprintln!("Error analyzing exercise vs scores: {}", e);
            }
            
            println!("\nAnalysis complete! Check the generated PNG files for visualizations.");
        }
        Err(e) => {
            eprintln!("Error: Failed to load students: {}", e);
            if let Some(source) = e.source() {
                eprintln!("Caused by: {}", source);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Helper function to create a temporary CSV file for testing
    fn create_test_csv() -> NamedTempFile {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(
            file,
            "study_hours_per_day,sleep_hours,mental_health_rating,exam_score,social_media_hours,netflix_hours,attendance_percentage,exercise_frequency"
        ).unwrap();
        writeln!(
            file,
            "2.5,8,7,85,2,1,95,Regular"
        ).unwrap();
        writeln!(
            file,
            "3.0,7,8,90,1,0.5,98,Regular"
        ).unwrap();
        writeln!(
            file,
            "1.5,6,5,70,4,2,80,Occasional"
        ).unwrap();
        file
    }

    #[test]
    fn test_load_students() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        assert_eq!(students.len(), 3);
        assert_eq!(students[0].study_hours_per_day, 2.5);
        assert_eq!(students[0].sleep_hours, 8.0);
        assert_eq!(students[0].mental_health_rating, 7);
        assert_eq!(students[0].exam_score, 85.0);
    }

    #[test]
    fn test_calculate_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let correlation = calculate_correlation(&x, &y);
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let normalized = normalize(&data);
        
        assert_eq!(normalized.len(), 5);
        assert!((normalized[0] - 0.0).abs() < 1e-10);
        assert!((normalized[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_analyze_study_hours_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_study_hours_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_sleep_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_sleep_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_mental_health_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_mental_health_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_social_media_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_social_media_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_attendance_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_attendance_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_exercise_vs_scores() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = analyze_exercise_vs_scores(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perform_kmeans_analysis() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = perform_kmeans_analysis(&students);
        assert!(result.is_ok());
    }

    #[test]
    fn test_perform_linear_regression() {
        let file = create_test_csv();
        let students = load_students(file.path()).unwrap();
        
        let result = perform_linear_regression(&students);
        assert!(result.is_ok());
    }
}
