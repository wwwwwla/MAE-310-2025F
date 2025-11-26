clear; clc; close all;

exact = @(x) sin(x);
exact_x = @(x) cos(x);
kappa = @(x) 1 + 0.5*x;
f = @(x) sin(x) + 0.5*x.*sin(x) - 0.5*cos(x);
g = sin(1);
h = -1;

% 2b
element_counts = [2, 4, 6, 8, 10, 12, 14, 16];
L2_errors_linear = [];
H1_errors_linear = [];
h_values_linear = [];

for n_el = element_counts
    n_np = n_el + 1;
    hh = 1 / n_el;
    x_coor = 0:hh:1;
    
    IEN = zeros(2, n_el);
    for ee = 1:n_el
        IEN(1, ee) = ee;
        IEN(2, ee) = ee + 1;
    end
    
    K_global = zeros(n_np, n_np);
    F_global = zeros(n_np, 1);
    
    xi = [-1/sqrt(3), 1/sqrt(3)];
    weight = [1, 1];
    
    for ee = 1:n_el
        k_ele = zeros(2, 2);
        f_ele = zeros(2, 1);
        
        x1 = x_coor(IEN(1, ee));
        x2 = x_coor(IEN(2, ee));
        L = x2 - x1;
        
        for qq = 1:2
            if qq == 1
                N1 = 0.5 * (1 - xi(1));
                N2 = 0.5 * (1 + xi(1));
                dN1_dxi = -0.5;
                dN2_dxi = 0.5;
            else
                N1 = 0.5 * (1 - xi(2));
                N2 = 0.5 * (1 + xi(2));
                dN1_dxi = -0.5;
                dN2_dxi = 0.5;
            end
            
            dx_dxi = 0.5 * L;
            dxi_dx = 2 / L;
            x_point = x1 * N1 + x2 * N2;
            kappa_val = kappa(x_point);
            
            f_ele(1) = f_ele(1) + weight(qq) * N1 * f(x_point) * dx_dxi;
            f_ele(2) = f_ele(2) + weight(qq) * N2 * f(x_point) * dx_dxi;
            
            k_ele(1,1) = k_ele(1,1) + weight(qq) * kappa_val * dN1_dxi * dN1_dxi * dxi_dx;
            k_ele(1,2) = k_ele(1,2) + weight(qq) * kappa_val * dN1_dxi * dN2_dxi * dxi_dx;
            k_ele(2,1) = k_ele(2,1) + weight(qq) * kappa_val * dN2_dxi * dN1_dxi * dxi_dx;
            k_ele(2,2) = k_ele(2,2) + weight(qq) * kappa_val * dN2_dxi * dN2_dxi * dxi_dx;
        end
        
        nodes = IEN(:, ee);
        K_global(nodes, nodes) = K_global(nodes, nodes) + k_ele;
        F_global(nodes) = F_global(nodes) + f_ele;
    end
    
    F_global(1) = F_global(1) + h;

    dirichlet_node = n_np;

    for i = 1:n_np
        if i == dirichlet_node
            continue;
        end
        F_global(i) = F_global(i) - K_global(i, dirichlet_node) * g;
    end
    
    K_global(dirichlet_node, :) = 0;
    K_global(:, dirichlet_node) = 0;
    K_global(dirichlet_node, dirichlet_node) = 1;
    F_global(dirichlet_node) = g;

    reg_param = 1e-12;
    K_global = K_global + reg_param * eye(n_np);

    disp = K_global \ F_global;
    
    [L2_err, H1_err] = calculate_errors(disp, x_coor, IEN, exact, exact_x);
    
    L2_errors_linear = [L2_errors_linear, L2_err];
    H1_errors_linear = [H1_errors_linear, H1_err];
    h_values_linear = [h_values_linear, hh];
end

% 2c
L2_errors_quad = [];
H1_errors_quad = [];
h_values_quad = [];

for n_el = [2, 4, 6, 8, 10, 12]
    n_np = 2 * n_el + 1;
    hh = 1 / n_el;
    
    x_coor = zeros(1, n_np);
    for i = 1:n_np
        x_coor(i) = (i-1) * hh / 2;
    end
    
    IEN = zeros(3, n_el);
    for ee = 1:n_el
        IEN(1, ee) = 2*(ee-1) + 1;
        IEN(2, ee) = 2*(ee-1) + 2;
        IEN(3, ee) = 2*(ee-1) + 3;
    end
    
    K_global = zeros(n_np, n_np);
    F_global = zeros(n_np, 1);
    
    xi = [-sqrt(3/5), 0, sqrt(3/5)];
    weight = [5/9, 8/9, 5/9];
    
    for ee = 1:n_el
        k_ele = zeros(3, 3);
        f_ele = zeros(3, 1);
        
        x1 = x_coor(IEN(1, ee));
        x2 = x_coor(IEN(2, ee));
        x3 = x_coor(IEN(3, ee));
        
        for qq = 1:3
            xi_val = xi(qq);
            
            N1 = 0.5 * xi_val * (xi_val - 1);
            N2 = 1 - xi_val^2;
            N3 = 0.5 * xi_val * (xi_val + 1);

            dN1_dxi = xi_val - 0.5;
            dN2_dxi = -2 * xi_val;
            dN3_dxi = xi_val + 0.5;
            
            dx_dxi = x1 * dN1_dxi + x2 * dN2_dxi + x3 * dN3_dxi;
            dxi_dx = 1 / dx_dxi;
            
            x_point = x1 * N1 + x2 * N2 + x3 * N3;
            kappa_val = kappa(x_point);
            
            f_ele(1) = f_ele(1) + weight(qq) * N1 * f(x_point) * dx_dxi;
            f_ele(2) = f_ele(2) + weight(qq) * N2 * f(x_point) * dx_dxi;
            f_ele(3) = f_ele(3) + weight(qq) * N3 * f(x_point) * dx_dxi;
            
            B = [dN1_dxi, dN2_dxi, dN3_dxi];
            k_ele = k_ele + weight(qq) * kappa_val * (B' * B) * dxi_dx;
        end
        
        nodes = IEN(:, ee);
        K_global(nodes, nodes) = K_global(nodes, nodes) + k_ele;
        F_global(nodes) = F_global(nodes) + f_ele;
    end

    F_global(1) = F_global(1) + h;
    
    dirichlet_node = n_np;
    for i = 1:n_np
        if i == dirichlet_node
            continue;
        end
        F_global(i) = F_global(i) - K_global(i, dirichlet_node) * g;
    end
    
    K_global(dirichlet_node, :) = 0;
    K_global(:, dirichlet_node) = 0;
    K_global(dirichlet_node, dirichlet_node) = 1;
    F_global(dirichlet_node) = g;
    
    reg_param = 1e-12;
    K_global = K_global + reg_param * eye(n_np);
    
    disp = K_global \ F_global;
    
    [L2_err, H1_err] = calculate_errors(disp, x_coor, IEN, exact, exact_x);
    
    L2_errors_quad = [L2_errors_quad, L2_err];
    H1_errors_quad = [H1_errors_quad, H1_err];
    h_values_quad = [h_values_quad, hh];
end

% 2c
L2_errors_cubic = [];
H1_errors_cubic = [];
h_values_cubic = [];

for n_el = [2, 4, 6, 8]
    n_np = 3 * n_el + 1;
    hh = 1 / n_el;
    
    x_coor = zeros(1, n_np);
    for i = 1:n_np
        x_coor(i) = (i-1) * hh / 3;
    end
    
    IEN = zeros(4, n_el);
    for ee = 1:n_el
        IEN(1, ee) = 3*(ee-1) + 1;
        IEN(2, ee) = 3*(ee-1) + 2;
        IEN(3, ee) = 3*(ee-1) + 3;
        IEN(4, ee) = 3*(ee-1) + 4;
    end
    
    K_global = zeros(n_np, n_np);
    F_global = zeros(n_np, 1);

    xi = [-0.861136, -0.339981, 0.339981, 0.861136];
    weight = [0.347855, 0.652145, 0.652145, 0.347855];
    
    for ee = 1:n_el
        k_ele = zeros(4, 4);
        f_ele = zeros(4, 1);
        
        x1 = x_coor(IEN(1, ee));
        x2 = x_coor(IEN(2, ee));
        x3 = x_coor(IEN(3, ee));
        x4 = x_coor(IEN(4, ee));
        
        for qq = 1:4
            xi_val = xi(qq);

            N1 = (1/16)*(1 - xi_val)*(-1 + 9*xi_val^2);
            N2 = (9/16)*(1 - xi_val^2)*(1 - 3*xi_val);
            N3 = (9/16)*(1 - xi_val^2)*(1 + 3*xi_val);
            N4 = (1/16)*(1 + xi_val)*(-1 + 9*xi_val^2);

            dN1_dxi = (1/16)*(-27*xi_val^2 + 18*xi_val + 1);
            dN2_dxi = (9/16)*(9*xi_val^2 - 2*xi_val - 3);
            dN3_dxi = (9/16)*(-9*xi_val^2 - 2*xi_val + 3);
            dN4_dxi = (1/16)*(27*xi_val^2 + 18*xi_val - 1);
            
            dx_dxi = x1 * dN1_dxi + x2 * dN2_dxi + x3 * dN3_dxi + x4 * dN4_dxi;
            dxi_dx = 1 / dx_dxi;
            
            x_point = x1 * N1 + x2 * N2 + x3 * N3 + x4 * N4;
            kappa_val = kappa(x_point);
            
            f_ele(1) = f_ele(1) + weight(qq) * N1 * f(x_point) * dx_dxi;
            f_ele(2) = f_ele(2) + weight(qq) * N2 * f(x_point) * dx_dxi;
            f_ele(3) = f_ele(3) + weight(qq) * N3 * f(x_point) * dx_dxi;
            f_ele(4) = f_ele(4) + weight(qq) * N4 * f(x_point) * dx_dxi;
            
            B = [dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi];
            k_ele = k_ele + weight(qq) * kappa_val * (B' * B) * dxi_dx;
        end
        
        nodes = IEN(:, ee);
        K_global(nodes, nodes) = K_global(nodes, nodes) + k_ele;
        F_global(nodes) = F_global(nodes) + f_ele;
    end

    F_global(1) = F_global(1) + h;
    
    dirichlet_node = n_np;
    for i = 1:n_np
        if i == dirichlet_node
            continue;
        end
        F_global(i) = F_global(i) - K_global(i, dirichlet_node) * g;
    end
    
    K_global(dirichlet_node, :) = 0;
    K_global(:, dirichlet_node) = 0;
    K_global(dirichlet_node, dirichlet_node) = 1;
    F_global(dirichlet_node) = g;
    
    reg_param = 1e-12;
    K_global = K_global + reg_param * eye(n_np);
    
    disp = K_global \ F_global;
    
    [L2_err, H1_err] = calculate_errors(disp, x_coor, IEN, exact, exact_x);
    
    L2_errors_cubic = [L2_errors_cubic, L2_err];
    H1_errors_cubic = [H1_errors_cubic, H1_err];
    h_values_cubic = [h_values_cubic, hh];
end

figure(1);
loglog(h_values_linear, L2_errors_linear, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
hold on;
loglog(h_values_linear, H1_errors_linear, 'rs-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(h_values_quad, L2_errors_quad, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(h_values_quad, H1_errors_quad, 'bs-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(h_values_cubic, L2_errors_cubic, 'go-', 'LineWidth', 2, 'MarkerSize', 8);
loglog(h_values_cubic, H1_errors_cubic, 'gs-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Mesh size h');
ylabel('Relative error');
legend('L2 linear', 'H1 linear', 'L2 quadratic', 'H1 quadratic', 'L2 cubic', 'H1 cubic', 'Location', 'best');
grid on;
title('Convergence Analysis for Different Element Types');

% Calculate convergence rates
fprintf('Linear Elements:\n');
calculate_convergence_rate(h_values_linear, L2_errors_linear, 'L2');
calculate_convergence_rate(h_values_linear, H1_errors_linear, 'H1');

fprintf('\nQuadratic Elements:\n');
calculate_convergence_rate(h_values_quad, L2_errors_quad, 'L2');
calculate_convergence_rate(h_values_quad, H1_errors_quad, 'H1');

fprintf('\nCubic Elements:\n');
calculate_convergence_rate(h_values_cubic, L2_errors_cubic, 'L2');
calculate_convergence_rate(h_values_cubic, H1_errors_cubic, 'H1');

% 2d
n_el = 8;  
quad_points = [2, 3, 4, 5, 6];
L2_quad_study_cubic = [];
H1_quad_study_cubic = [];

for n_int = quad_points
    n_np = 3 * n_el + 1;
    hh = 1 / n_el;
    
    x_coor = zeros(1, n_np);
    for i = 1:n_np
        x_coor(i) = (i-1) * hh / 3;
    end
    
    IEN = zeros(4, n_el);
    for ee = 1:n_el
        IEN(1, ee) = 3*(ee-1) + 1;
        IEN(2, ee) = 3*(ee-1) + 2;
        IEN(3, ee) = 3*(ee-1) + 3;
        IEN(4, ee) = 3*(ee-1) + 4;
    end
    
    K_global = zeros(n_np, n_np);
    F_global = zeros(n_np, 1);
    
    if n_int == 2
        xi = [-1/sqrt(3), 1/sqrt(3)];
        weight = [1, 1];
    elseif n_int == 3
        xi = [-sqrt(3/5), 0, sqrt(3/5)];
        weight = [5/9, 8/9, 5/9];
    elseif n_int == 4
        xi = [-0.861136, -0.339981, 0.339981, 0.861136];
        weight = [0.347855, 0.652145, 0.652145, 0.347855];
    elseif n_int == 5
        xi = [-0.906180, -0.538469, 0, 0.538469, 0.906180];
        weight = [0.236927, 0.478629, 0.568889, 0.478629, 0.236927];
    elseif n_int == 6
        xi = [-0.932470, -0.661209, -0.238619, 0.238619, 0.661209, 0.932470];
        weight = [0.171324, 0.360762, 0.467914, 0.467914, 0.360762, 0.171324];
    end
    
    for ee = 1:n_el
        k_ele = zeros(4, 4);
        f_ele = zeros(4, 1);
        
        x1 = x_coor(IEN(1, ee));
        x2 = x_coor(IEN(2, ee));
        x3 = x_coor(IEN(3, ee));
        x4 = x_coor(IEN(4, ee));
        
        for qq = 1:n_int
            xi_val = xi(qq);

            N1 = (1/16)*(1 - xi_val)*(-1 + 9*xi_val^2);
            N2 = (9/16)*(1 - xi_val^2)*(1 - 3*xi_val);
            N3 = (9/16)*(1 - xi_val^2)*(1 + 3*xi_val);
            N4 = (1/16)*(1 + xi_val)*(-1 + 9*xi_val^2);
            
            dN1_dxi = (1/16)*(-27*xi_val^2 + 18*xi_val + 1);
            dN2_dxi = (9/16)*(9*xi_val^2 - 2*xi_val - 3);
            dN3_dxi = (9/16)*(-9*xi_val^2 - 2*xi_val + 3);
            dN4_dxi = (1/16)*(27*xi_val^2 + 18*xi_val - 1);
            
            dx_dxi = x1 * dN1_dxi + x2 * dN2_dxi + x3 * dN3_dxi + x4 * dN4_dxi;
            
            if abs(dx_dxi) < 1e-10
                continue;
            end
            
            dxi_dx = 1 / dx_dxi;
            x_point = x1 * N1 + x2 * N2 + x3 * N3 + x4 * N4;
            kappa_val = kappa(x_point);
            
            f_ele(1) = f_ele(1) + weight(qq) * N1 * f(x_point) * dx_dxi;
            f_ele(2) = f_ele(2) + weight(qq) * N2 * f(x_point) * dx_dxi;
            f_ele(3) = f_ele(3) + weight(qq) * N3 * f(x_point) * dx_dxi;
            f_ele(4) = f_ele(4) + weight(qq) * N4 * f(x_point) * dx_dxi;
            
            B = [dN1_dxi, dN2_dxi, dN3_dxi, dN4_dxi];
            k_ele = k_ele + weight(qq) * kappa_val * (B' * B) * dxi_dx;
        end
        
        nodes = IEN(:, ee);
        K_global(nodes, nodes) = K_global(nodes, nodes) + k_ele;
        F_global(nodes) = F_global(nodes) + f_ele;
    end

    F_global(1) = F_global(1) + h;
    
    dirichlet_node = n_np;

    for i = 1:n_np
        if i ~= dirichlet_node
            F_global(i) = F_global(i) - K_global(i, dirichlet_node) * g;
        end
    end

    K_global(dirichlet_node, :) = 0;
    K_global(:, dirichlet_node) = 0;
    K_global(dirichlet_node, dirichlet_node) = 1;
    F_global(dirichlet_node) = g;

    reg_param = 1e-8 * norm(K_global, 'fro');
    K_global = K_global + reg_param * eye(n_np);

    disp = K_global \ F_global;
    
    [L2_err, H1_err] = calculate_errors(disp, x_coor, IEN, exact, exact_x);
    
    L2_quad_study_cubic = [L2_quad_study_cubic, L2_err];
    H1_quad_study_cubic = [H1_quad_study_cubic, H1_err];
    
    fprintf('Quadrature points: %d, L2 error: %.6e, H1 error: %.6e\n', ...
            n_int, L2_err, H1_err);
end

figure(2);
subplot(2,1,1);
semilogy(quad_points, L2_quad_study_cubic, 'bo-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of quadrature points');
ylabel('L2 error');
title('Cubic Element: Effect of Quadrature Points on L2 Error');
grid on;

subplot(2,1,2);
semilogy(quad_points, H1_quad_study_cubic, 'ro-', 'LineWidth', 2, 'MarkerSize', 8);
xlabel('Number of quadrature points');
ylabel('H1 error');
title('Cubic Element: Effect of Quadrature Points on H1 Error');
grid on;

%
function [L2_error, H1_error] = calculate_errors(disp, x_coor, IEN, exact, exact_x)
    n_el = size(IEN, 2);
    n_en = size(IEN, 1);
    
    xi = [-0.861136, -0.339981, 0.339981, 0.861136];
    weight = [0.347855, 0.652145, 0.652145, 0.347855];
    
    L2_top = 0;
    L2_bot = 0;
    H1_top = 0;
    H1_bot = 0;
    
    for ee = 1:n_el
        nodes = IEN(:, ee);
        x_ele = x_coor(nodes);
        u_ele = disp(nodes);
        
        for qq = 1:4
            x_val = 0;
            dx_dxi = 0;
            uh = 0;
            duh_dxi = 0;
            
            for aa = 1:n_en
                if n_en == 2 
                    if aa == 1
                        N = 0.5 * (1 - xi(qq));
                        dN_dxi = -0.5;
                    else
                        N = 0.5 * (1 + xi(qq));
                        dN_dxi = 0.5;
                    end
                elseif n_en == 3  
                    if aa == 1
                        N = 0.5 * xi(qq) * (xi(qq) - 1);
                        dN_dxi = xi(qq) - 0.5;
                    elseif aa == 2
                        N = 1 - xi(qq)^2;
                        dN_dxi = -2 * xi(qq);
                    else
                        N = 0.5 * xi(qq) * (xi(qq) + 1);
                        dN_dxi = xi(qq) + 0.5;
                    end
                else 
                    if aa == 1
                        N = (1/16)*(1 - xi(qq))*(-1 + 9*xi(qq)^2);
                        dN_dxi = (1/16)*(-27*xi(qq)^2 + 18*xi(qq) + 1);
                    elseif aa == 2
                        N = (9/16)*(1 - xi(qq)^2)*(1 - 3*xi(qq));
                        dN_dxi = (9/16)*(9*xi(qq)^2 - 2*xi(qq) - 3);
                    elseif aa == 3
                        N = (9/16)*(1 - xi(qq)^2)*(1 + 3*xi(qq));
                        dN_dxi = (9/16)*(-9*xi(qq)^2 - 2*xi(qq) + 3);
                    else
                        N = (1/16)*(1 + xi(qq))*(-1 + 9*xi(qq)^2);
                        dN_dxi = (1/16)*(27*xi(qq)^2 + 18*xi(qq) - 1);
                    end
                end
                
                x_val = x_val + x_ele(aa) * N;
                dx_dxi = dx_dxi + x_ele(aa) * dN_dxi;
                uh = uh + u_ele(aa) * N;
                duh_dxi = duh_dxi + u_ele(aa) * dN_dxi;
            end
            
            dxi_dx = 1 / dx_dxi;
            duh_dx = duh_dxi * dxi_dx;
            
            exact_val = exact(x_val);
            exact_x_val = exact_x(x_val);
            
            L2_top = L2_top + weight(qq) * (uh - exact_val)^2 * dx_dxi;
            L2_bot = L2_bot + weight(qq) * exact_val^2 * dx_dxi;
            H1_top = H1_top + weight(qq) * (duh_dx - exact_x_val)^2 * dx_dxi;
            H1_bot = H1_bot + weight(qq) * exact_x_val^2 * dx_dxi;
        end
    end
    
    L2_error = sqrt(L2_top) / sqrt(L2_bot);
    H1_error = sqrt(H1_top) / sqrt(H1_bot);
end

function calculate_convergence_rate(h_values, errors, error_type)
    n = length(h_values);
    rates = zeros(1, n-1);
    
    for i = 1:n-1
        rates(i) = log(errors(i+1)/errors(i)) / log(h_values(i+1)/h_values(i));
    end
    
    avg_rate = mean(rates);
    fprintf('%s error convergence rate: %.3f\n', error_type, avg_rate);
end